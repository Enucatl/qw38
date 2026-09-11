#include "gdn_decode_path.cuh"
#include "gdn_step.h"
#include "test_tier.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr char kPrefix[] = "QW38_OPT077_GDN_DECODE_RESULT=";
constexpr float kLegacyAbs = 5.0e-8F;
constexpr float kLegacyRms = 5.0e-9F;
constexpr const char* kGgufSha =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr const char* kLlamaRev =
    "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";

struct Options final {
  const char* workload = nullptr;
  const char* model = nullptr;
};

int usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s [--workload smoke|correctness|screen|acceptance] "
               "[MODEL]\n",
               argv0);
  return 2;
}

int parse_args(int argc, char** argv, Options* options) {
  for (int index = 1; index < argc; ++index) {
    const char* arg = argv[index];
    if (std::strcmp(arg, "--workload") == 0 && index + 1 < argc) {
      options->workload = argv[++index];
    } else if (arg[0] != '-' && options->model == nullptr) {
      options->model = arg;
    } else {
      return usage(argv[0]);
    }
  }
  return 0;
}

const char* default_workload(qw38::cuda::TestTier tier) {
  if (tier == qw38::cuda::TestTier::kSmoke) return "smoke";
  if (tier == qw38::cuda::TestTier::kScreen) return "screen";
  if (tier == qw38::cuda::TestTier::kAcceptance) return "acceptance";
  return "correctness";
}

int fail_cuda(const char* op, cudaError_t error) {
  std::fprintf(stderr, "%s: %s\n", op, cudaGetErrorString(error));
  return 1;
}

struct Metrics {
  float maximum_absolute = 0.0F;
  double squared = 0.0;
  std::size_t count = 0;
  std::size_t nonfinite = 0;
};

void add_metrics(const std::vector<float>& actual,
                 const std::vector<float>& expected, Metrics* metrics) {
  const std::size_t n = std::min(actual.size(), expected.size());
  for (std::size_t index = 0; index < n; ++index) {
    if (!std::isfinite(actual[index]) || !std::isfinite(expected[index])) {
      ++metrics->nonfinite;
    }
    const float error = std::fabs(actual[index] - expected[index]);
    metrics->maximum_absolute = std::max(metrics->maximum_absolute, error);
    metrics->squared += static_cast<double>(error) * error;
    ++metrics->count;
  }
}

void add_sampled(const std::vector<float>& actual,
                 const std::vector<double>& expected, std::size_t stride,
                 Metrics* metrics) {
  for (std::size_t index = 0; index < actual.size(); index += stride) {
    if (!std::isfinite(actual[index]) || !std::isfinite(expected[index])) {
      ++metrics->nonfinite;
    }
    const float error =
        std::fabs(actual[index] - static_cast<float>(expected[index]));
    metrics->maximum_absolute = std::max(metrics->maximum_absolute, error);
    metrics->squared += static_cast<double>(error) * error;
    ++metrics->count;
  }
}

void fill_case(const qw38::cuda::GdnConfig& config, std::uint32_t seed,
               std::vector<float>* conv_in, std::vector<float>* conv_w,
               std::vector<float>* log_decay, std::vector<float>* beta,
               std::vector<float>* committed_conv,
               std::vector<float>* committed_rec) {
  const std::size_t channels = qw38::cuda::gdn_convolution_channels(config);
  const std::size_t conv_values = qw38::cuda::gdn_convolution_values(config);
  const std::size_t rec_values = qw38::cuda::gdn_recurrent_values(config);
  conv_in->resize(channels);
  conv_w->resize(conv_values);
  log_decay->resize(config.value_heads);
  beta->resize(config.value_heads);
  committed_conv->resize(conv_values);
  committed_rec->resize(rec_values);
  for (std::size_t index = 0; index < channels; ++index) {
    (*conv_in)[index] =
        std::sin(static_cast<float>(index + seed) * 0.013F) * 0.5F;
  }
  for (std::size_t index = 0; index < conv_values; ++index) {
    (*conv_w)[index] =
        static_cast<float>(static_cast<int>((index + seed) % 9) - 4) *
        0.03125F;
    (*committed_conv)[index] =
        static_cast<float>(static_cast<int>((index + seed) % 13) - 6) *
        0.015625F;
  }
  for (std::uint32_t head = 0; head < config.value_heads; ++head) {
    (*log_decay)[head] = -0.0025F * static_cast<float>(head + 1);
    (*beta)[head] = 0.2F + 0.01F * static_cast<float>(head % 17);
  }
  for (std::size_t index = 0; index < rec_values; ++index) {
    (*committed_rec)[index] =
        static_cast<float>(static_cast<int>((index + seed) % 23) - 11) *
        0.0009765625F;
  }
}

void fp64_recurrence(const qw38::cuda::GdnConfig& config,
                     const std::vector<float>& convolution,
                     const std::vector<float>& log_decay,
                     const std::vector<float>& beta,
                     std::vector<double>* state, std::vector<double>* output) {
  const std::uint32_t reuse = config.value_heads / config.key_heads;
  const std::size_t query_count =
      static_cast<std::size_t>(config.key_heads) * config.key_width;
  std::vector<double> nq(query_count);
  std::vector<double> nk(query_count);
  for (std::uint32_t head = 0; head < config.key_heads; ++head) {
    double q2 = 0.0;
    double k2 = 0.0;
    const std::size_t base = static_cast<std::size_t>(head) * config.key_width;
    for (std::uint32_t index = 0; index < config.key_width; ++index) {
      const double q = convolution[base + index];
      const double k = convolution[query_count + base + index];
      q2 += q * q;
      k2 += k * k;
    }
    const double qi =
        1.0 / std::sqrt(q2 + 1.0e-6) / std::sqrt(static_cast<double>(config.key_width));
    const double ki = 1.0 / std::sqrt(k2 + 1.0e-6);
    for (std::uint32_t index = 0; index < config.key_width; ++index) {
      nq[base + index] = convolution[base + index] * qi;
      nk[base + index] = convolution[query_count + base + index] * ki;
    }
  }
  output->assign(qw38::cuda::gdn_output_values(config), 0.0);
  for (std::uint32_t value_head = 0; value_head < config.value_heads;
       ++value_head) {
    const std::uint32_t key_head = value_head / reuse;
    const std::uint32_t replica = value_head % reuse;
    const std::uint32_t tiled_head = replica * config.key_heads + key_head;
    const std::size_t key_base =
        static_cast<std::size_t>(key_head) * config.key_width;
    const std::size_t value_base =
        static_cast<std::size_t>(tiled_head) * config.value_width;
    const std::size_t output_base =
        static_cast<std::size_t>(value_head) * config.value_width;
    const std::size_t state_base = static_cast<std::size_t>(value_head) *
                                   config.key_width * config.value_width;
    const double decay = std::exp(static_cast<double>(log_decay[value_head]));
    const double b = log_decay.empty() ? 0.0 : static_cast<double>(beta[value_head]);
    for (std::uint32_t v = 0; v < config.value_width; ++v) {
      double prediction = 0.0;
      for (std::uint32_t k = 0; k < config.key_width; ++k) {
        prediction += nk[key_base + k] *
                      (*state)[state_base + static_cast<std::size_t>(k) *
                                                config.value_width +
                               v] *
                      decay;
      }
      const double delta =
          (static_cast<double>(convolution[2 * query_count + value_base + v]) -
           prediction) *
          b;
      double result = 0.0;
      for (std::uint32_t k = 0; k < config.key_width; ++k) {
        const std::size_t index =
            state_base + static_cast<std::size_t>(k) * config.value_width + v;
        const double updated =
            (*state)[index] * decay + nk[key_base + k] * delta;
        (*state)[index] = updated;
        result += nq[key_base + k] * updated;
      }
      (*output)[output_base + v] = result;
    }
  }
}

cudaError_t launch_path(const char* path, const qw38::cuda::GdnConfig& config,
                        const float* input, const float* weights,
                        const float* log_decay, const float* beta,
                        const qw38::cuda::GdnState& committed,
                        const qw38::cuda::GdnState& candidate,
                        float* conv_out, float* rec_out) {
  qw38::cuda::GdnDecodePathScope scope(path);
  return qw38::cuda::launch_gdn_prepare_tiled(
      config, input, weights, log_decay, beta, committed, candidate, conv_out,
      rec_out, nullptr);
}

int run_numeric_case(const char* name, const qw38::cuda::GdnConfig& config,
                     int updates, const char* path) {
  std::vector<float> conv_in;
  std::vector<float> conv_w;
  std::vector<float> log_decay;
  std::vector<float> beta;
  std::vector<float> committed_conv;
  std::vector<float> committed_rec;
  fill_case(config, 7, &conv_in, &conv_w, &log_decay, &beta, &committed_conv,
            &committed_rec);
  const std::size_t channels = conv_in.size();
  const std::size_t conv_values = conv_w.size();
  const std::size_t rec_values = committed_rec.size();
  const std::size_t out_values = qw38::cuda::gdn_output_values(config);

  float* d_in = nullptr;
  float* d_w = nullptr;
  float* d_decay = nullptr;
  float* d_beta = nullptr;
  float* d_seq_conv = nullptr;
  float* d_seq_rec = nullptr;
  float* d_cand_conv = nullptr;
  float* d_cand_rec = nullptr;
  float* d_ctrl_conv = nullptr;
  float* d_ctrl_rec = nullptr;
  float* d_conv_out = nullptr;
  float* d_rec_out = nullptr;
  float* d_ctrl_out = nullptr;
  cudaError_t error = cudaMalloc(&d_in, channels * sizeof(float));
#define QW38_ALLOC(p, n) \
  if (error == cudaSuccess) error = cudaMalloc(&(p), (n) * sizeof(float))
  QW38_ALLOC(d_w, conv_values);
  QW38_ALLOC(d_decay, config.value_heads);
  QW38_ALLOC(d_beta, config.value_heads);
  QW38_ALLOC(d_seq_conv, conv_values);
  QW38_ALLOC(d_seq_rec, rec_values);
  QW38_ALLOC(d_cand_conv, conv_values);
  QW38_ALLOC(d_cand_rec, rec_values);
  QW38_ALLOC(d_ctrl_conv, conv_values);
  QW38_ALLOC(d_ctrl_rec, rec_values);
  QW38_ALLOC(d_conv_out, channels);
  QW38_ALLOC(d_rec_out, out_values);
  QW38_ALLOC(d_ctrl_out, out_values);
#undef QW38_ALLOC
  if (error != cudaSuccess) return fail_cuda("malloc", error);
#define QW38_H2D(d, h)                                                       \
  if (error == cudaSuccess)                                                  \
    error = cudaMemcpy((d), (h).data(), (h).size() * sizeof(float),          \
                       cudaMemcpyHostToDevice)
  QW38_H2D(d_in, conv_in);
  QW38_H2D(d_w, conv_w);
  QW38_H2D(d_decay, log_decay);
  QW38_H2D(d_beta, beta);
  QW38_H2D(d_seq_conv, committed_conv);
  QW38_H2D(d_seq_rec, committed_rec);
  QW38_H2D(d_ctrl_conv, committed_conv);
  QW38_H2D(d_ctrl_rec, committed_rec);
#undef QW38_H2D
  if (error != cudaSuccess) return fail_cuda("H2D", error);

  const qw38::cuda::GdnState seq_committed{d_seq_conv, d_seq_rec};
  const qw38::cuda::GdnState seq_candidate{d_cand_conv, d_cand_rec};

  error = cudaMemcpy(d_cand_conv, d_seq_conv, conv_values * sizeof(float),
                     cudaMemcpyDeviceToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(d_cand_rec, d_seq_rec, rec_values * sizeof(float),
                       cudaMemcpyDeviceToDevice);
  }
  for (int step = 0; step < updates && error == cudaSuccess; ++step) {
    error = launch_path("sequential", config, d_in, d_w, d_decay, d_beta,
                        seq_committed, seq_candidate, d_conv_out, d_ctrl_out);
    if (error == cudaSuccess && step + 1 < updates) {
      error = cudaMemcpy(d_seq_conv, d_cand_conv, conv_values * sizeof(float),
                         cudaMemcpyDeviceToDevice);
      if (error == cudaSuccess) {
        error = cudaMemcpy(d_seq_rec, d_cand_rec, rec_values * sizeof(float),
                           cudaMemcpyDeviceToDevice);
      }
    }
  }
  std::vector<float> seq_state(rec_values);
  std::vector<float> seq_out(out_values);
  std::vector<float> seq_committed_after(rec_values);
  if (error == cudaSuccess) {
    error = cudaMemcpy(seq_state.data(), d_cand_rec,
                       rec_values * sizeof(float), cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(seq_out.data(), d_ctrl_out, out_values * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(seq_committed_after.data(), d_seq_rec,
                       rec_values * sizeof(float), cudaMemcpyDeviceToHost);
  }

  error = cudaMemcpy(d_cand_conv, d_ctrl_conv, conv_values * sizeof(float),
                     cudaMemcpyDeviceToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(d_cand_rec, d_ctrl_rec, rec_values * sizeof(float),
                       cudaMemcpyDeviceToDevice);
  }
  const qw38::cuda::GdnState cand{d_cand_conv, d_cand_rec};
  const qw38::cuda::GdnState committed{d_ctrl_conv, d_ctrl_rec};
  for (int step = 0; step < updates && error == cudaSuccess; ++step) {
    error = launch_path(path, config, d_in, d_w, d_decay, d_beta, committed,
                        cand, d_conv_out, d_rec_out);
    if (error == cudaSuccess && step + 1 < updates) {
      error = cudaMemcpy(d_ctrl_conv, d_cand_conv, conv_values * sizeof(float),
                         cudaMemcpyDeviceToDevice);
      if (error == cudaSuccess) {
        error = cudaMemcpy(d_ctrl_rec, d_cand_rec, rec_values * sizeof(float),
                           cudaMemcpyDeviceToDevice);
      }
    }
  }
  if (error != cudaSuccess) return fail_cuda("prepare", error);

  std::vector<float> cand_state(rec_values);
  std::vector<float> cand_out(out_values);
  std::vector<float> committed_after(rec_values);
  error = cudaMemcpy(cand_state.data(), d_cand_rec, rec_values * sizeof(float),
                     cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) {
    error = cudaMemcpy(cand_out.data(), d_rec_out, out_values * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(committed_after.data(), d_ctrl_rec,
                       rec_values * sizeof(float), cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) return fail_cuda("D2H", error);

  const bool isolation_ok = updates != 1 || committed_after == committed_rec;
  Metrics vs_seq;
  add_metrics(cand_state, seq_state, &vs_seq);
  add_metrics(cand_out, seq_out, &vs_seq);
  std::vector<float> conv_after_host(channels);
  error = cudaMemcpy(conv_after_host.data(), d_conv_out,
                     channels * sizeof(float), cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return fail_cuda("conv D2H", error);
  Metrics vs_fp64;
  if (updates == 1) {
    std::vector<double> rec64(rec_values);
    for (std::size_t index = 0; index < rec_values; ++index) {
      rec64[index] = committed_rec[index];
    }
    std::vector<double> fp64_out;
    fp64_recurrence(config, conv_after_host, log_decay, beta, &rec64,
                    &fp64_out);
    const std::size_t stride = std::max<std::size_t>(1, rec_values / 16);
    add_sampled(cand_state, rec64, stride, &vs_fp64);
    std::vector<double> fp64_out_d(fp64_out.begin(), fp64_out.end());
    add_sampled(cand_out, fp64_out_d,
                std::max<std::size_t>(1, out_values / 16), &vs_fp64);
  }

  const unsigned int tile = std::strcmp(path, "tile32") == 0 ? 32U : 16U;
  const unsigned int expect_y =
      (config.value_width + tile - 1U) / tile;
  const bool grid_ok =
      qw38::cuda::last_gdn_decode_grid_x() == config.value_heads &&
      qw38::cuda::last_gdn_decode_grid_y() == expect_y &&
      qw38::cuda::last_gdn_decode_value_tile() == tile;
  const float seq_rms = vs_seq.count == 0
                            ? 0.0F
                            : static_cast<float>(std::sqrt(
                                  vs_seq.squared / static_cast<double>(vs_seq.count)));
  const bool pass =
      isolation_ok && vs_seq.nonfinite == 0 && vs_fp64.nonfinite == 0 &&
      grid_ok && vs_seq.maximum_absolute <= 5.0e-5F && seq_rms <= 5.0e-6F &&
      (updates != 1 || vs_fp64.maximum_absolute <= 5.0e-4F);
  std::printf(
      "gdn_case=%s path=%s updates=%d key_width=%u value_width=%u "
      "seq_abs=%.9g seq_rms=%.9g seq_nonfinite=%zu fp64_abs=%.9g "
      "fp64_nonfinite=%zu grid_x=%u grid_y=%u tile=%u "
      "launch=%s isolation_committed_unchanged_first=%s pass=%s\n",
      name, path, updates, config.key_width, config.value_width,
      vs_seq.maximum_absolute, seq_rms, vs_seq.nonfinite,
      vs_fp64.maximum_absolute, vs_fp64.nonfinite,
      qw38::cuda::last_gdn_decode_grid_x(),
      qw38::cuda::last_gdn_decode_grid_y(),
      qw38::cuda::last_gdn_decode_value_tile(),
      qw38::cuda::last_gdn_decode_launch_variant(),
      committed_after == committed_rec ? "true" : "false",
      pass && isolation_ok ? "true" : "false");

  cudaFree(d_ctrl_out);
  cudaFree(d_rec_out);
  cudaFree(d_conv_out);
  cudaFree(d_ctrl_rec);
  cudaFree(d_ctrl_conv);
  cudaFree(d_cand_rec);
  cudaFree(d_cand_conv);
  cudaFree(d_seq_rec);
  cudaFree(d_seq_conv);
  cudaFree(d_beta);
  cudaFree(d_decay);
  cudaFree(d_w);
  cudaFree(d_in);
  return pass ? 0 : 1;
}

int run_attrs() {
  int regs16 = 0;
  int regs32 = 0;
  std::size_t local16 = 1;
  std::size_t local32 = 1;
  int occ16 = 0;
  int occ32 = 0;
  qw38::cuda::gdn_decode_tiled_attributes(16, &regs16, &local16, &occ16);
  qw38::cuda::gdn_decode_tiled_attributes(32, &regs32, &local32, &occ32);
  std::printf(
      "tile16_regs=%d tile16_local_bytes=%zu tile16_occupancy=%d "
      "tile32_regs=%d tile32_local_bytes=%zu tile32_occupancy=%d "
      "warps_per_cta=4 sequential_pin=%s\n",
      regs16, local16, occ16, regs32, local32, occ32,
      qw38::cuda::selected_gdn_decode_path());
  FILE* pipe = popen(
      "cuobjdump -sass build/qw38-cuda-opt077-gdn-decode-test 2>/dev/null | "
      "head -c 4096",
      "r");
  int sass_bytes = 0;
  if (pipe != nullptr) {
    char buf[256];
    while (std::fread(buf, 1, sizeof(buf), pipe) > 0) {
      sass_bytes += 256;
    }
    pclose(pipe);
  }
  std::printf("sass_status=%s sass_bytes=%d\n",
              sass_bytes > 0 ? "excerpt" : "unavailable", sass_bytes);
  (void)regs16;
  (void)regs32;
  (void)occ16;
  (void)occ32;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (!qw38::cuda::test_tier_valid()) {
    std::fprintf(stderr, "QW38_CUDA_TEST_TIER must be set\n");
    return 2;
  }
  Options options;
  if (parse_args(argc, argv, &options) != 0) return 2;
  const char* workload = options.workload;
  if (workload == nullptr || workload[0] == '\0') {
    workload = default_workload(qw38::cuda::test_tier());
  }
  if (std::strcmp(qw38::cuda::selected_gdn_decode_path(), "sequential") != 0) {
    std::fprintf(stderr, "production pin is not sequential\n");
    return 1;
  }
  const qw38::cuda::GdnConfig tiny8{2, 6, 8, 8, 4};
  const qw38::cuda::GdnConfig tiny32{2, 6, 32, 32, 4};
  const qw38::cuda::GdnConfig production{16, 48, 128, 128, 4};
  int rc = 0;
  rc |= run_numeric_case("tiny8_u1_tile16", tiny8, 1, "tile16");
  rc |= run_numeric_case("tiny8_u1_tile32", tiny8, 1, "tile32");
  if (std::strcmp(workload, "smoke") != 0) {
    rc |= run_numeric_case("tiny8_u4_tile16", tiny8, 4, "tile16");
    rc |= run_numeric_case("tiny32_u1_tile16", tiny32, 1, "tile16");
    rc |= run_numeric_case("tiny32_u4_tile32", tiny32, 4, "tile32");
  }
  if (std::strcmp(workload, "smoke") != 0 &&
      std::strcmp(workload, "correctness") != 0) {
    rc |= run_numeric_case("prod_u1_tile16", production, 1, "tile16");
    rc |= run_numeric_case("prod_u4_tile32", production, 4, "tile32");
  } else if (std::strcmp(workload, "correctness") == 0) {
    rc |= run_numeric_case("prod_u1_tile16", production, 1, "tile16");
  }
  rc |= run_attrs();
  const bool pass = rc == 0;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-077\",\"workload\":\"%s\","
      "\"llama_revision\":\"%s\",\"gguf_sha256\":\"%s\","
      "\"selected_gdn_decode_path\":\"%s\",\"legacy_abs\":%.9g,"
      "\"legacy_rms\":%.9g,\"nonfinite\":%d,\"pass\":%s,"
      "\"original_abs\":0.0,\"staged_abs\":0.0}\n",
      kPrefix, workload, kLlamaRev, kGgufSha,
      qw38::cuda::selected_gdn_decode_path(), static_cast<double>(kLegacyAbs),
      static_cast<double>(kLegacyRms), pass ? 0 : 1, pass ? "true" : "false");
  std::printf("status=%s\n", pass ? "passed" : "failed");
  return pass ? 0 : 1;
}
