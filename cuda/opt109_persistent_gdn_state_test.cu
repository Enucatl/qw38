#include "gdn_decode_path.cuh"
#include "gdn_step.h"
#include "test_tier.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr char kPrefix[] = "QW38_OPT109_RESULT=";
constexpr char kCountsPrefix[] = "QW38_OPT109_NATIVE_COUNTS=";
constexpr char kCasePrefix[] = "QW38_OPT109_CASE=";
constexpr char kPilotPrefix[] = "QW38_OPT109_PILOT=";
constexpr float kSeqAbs = 5.0e-5F;
constexpr float kSeqRms = 5.0e-6F;
constexpr float kFp64Abs = 5.0e-4F;
constexpr const char* kGgufSha =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr const char* kLlamaRev =
    "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr std::size_t kSessionLayers = 48;

struct Options final {
  const char* workload = nullptr;
};

int usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s [--phase|--workload smoke|parity|correctness|"
               "state|pilot|screen|acceptance] [MODEL]\n",
               argv0);
  return 2;
}

int parse_args(int argc, char** argv, Options* options) {
  for (int index = 1; index < argc; ++index) {
    const char* arg = argv[index];
    if ((std::strcmp(arg, "--workload") == 0 ||
         std::strcmp(arg, "--phase") == 0) &&
        index + 1 < argc) {
      options->workload = argv[++index];
    } else if (arg[0] != '-') {
      continue;
    } else {
      return usage(argv[0]);
    }
  }
  return 0;
}

const char* default_workload(qw38::cuda::TestTier tier) {
  if (tier == qw38::cuda::TestTier::kSmoke) return "smoke";
  if (tier == qw38::cuda::TestTier::kScreen) return "pilot";
  if (tier == qw38::cuda::TestTier::kAcceptance) return "acceptance";
  return "parity";
}

int fail_cuda(const char* op, cudaError_t error) {
  std::fprintf(stderr, "%s: %s\n", op, cudaGetErrorString(error));
  return 1;
}

const char* json_bool(bool value) { return value ? "true" : "false"; }

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

void host_row_to_col(const std::vector<float>& row, std::vector<float>* col,
                     const qw38::cuda::GdnConfig& config) {
  col->assign(row.size(), 0.0F);
  for (std::uint32_t head = 0; head < config.value_heads; ++head) {
    for (std::uint32_t row_i = 0; row_i < config.key_width; ++row_i) {
      for (std::uint32_t col_i = 0; col_i < config.value_width; ++col_i) {
        (*col)[qw38::cuda::gdn_state_device_index(
            head, row_i, col_i, config.key_width, config.value_width)] =
            row[qw38::cuda::gdn_state_logical_index(
                head, row_i, col_i, config.key_width, config.value_width)];
      }
    }
  }
}

void host_col_to_row(const std::vector<float>& col, std::vector<float>* row,
                     const qw38::cuda::GdnConfig& config) {
  row->assign(col.size(), 0.0F);
  for (std::uint32_t head = 0; head < config.value_heads; ++head) {
    for (std::uint32_t row_i = 0; row_i < config.key_width; ++row_i) {
      for (std::uint32_t col_i = 0; col_i < config.value_width; ++col_i) {
        (*row)[qw38::cuda::gdn_state_logical_index(
            head, row_i, col_i, config.key_width, config.value_width)] =
            col[qw38::cuda::gdn_state_device_index(
                head, row_i, col_i, config.key_width, config.value_width)];
      }
    }
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
        1.0 / std::sqrt(q2 + 1.0e-6) /
        std::sqrt(static_cast<double>(config.key_width));
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
    const double b = static_cast<double>(beta[value_head]);
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

bool layout_roundtrip_host() {
  constexpr std::uint32_t heads = 2;
  constexpr std::uint32_t width = 8;
  std::vector<float> logical(heads * width * width);
  std::vector<float> device(logical.size());
  std::vector<float> restored(logical.size());
  for (std::uint32_t head = 0; head < heads; ++head) {
    for (std::uint32_t row = 0; row < width; ++row) {
      for (std::uint32_t col = 0; col < width; ++col) {
        const float value =
            static_cast<float>(head * 1000 + row * 20 + col);
        logical[qw38::cuda::gdn_state_logical_index(head, row, col, width,
                                                    width)] = value;
        device[qw38::cuda::gdn_state_device_index(head, row, col, width,
                                                  width)] = value;
      }
    }
  }
  for (std::uint32_t head = 0; head < heads; ++head) {
    for (std::uint32_t row = 0; row < width; ++row) {
      for (std::uint32_t col = 0; col < width; ++col) {
        restored[qw38::cuda::gdn_state_logical_index(head, row, col, width,
                                                     width)] =
            device[qw38::cuda::gdn_state_device_index(head, row, col, width,
                                                      width)];
      }
    }
  }
  const bool pass =
      logical.size() == device.size() && restored == logical && logical != device;
  std::printf(
      "%s{\"id\":\"layout_roundtrip\",\"pass\":%s,\"same_count\":%s,"
      "\"restored_ok\":%s,\"layouts_differ\":%s,"
      "\"logical\":\"%s\",\"device\":\"%s\",\"checkpoint\":\"%s\","
      "\"logical_sha256\":\"%s\",\"device_sha256\":\"%s\","
      "\"layout_version\":%u}\n",
      kCasePrefix, json_bool(pass), json_bool(logical.size() == device.size()),
      json_bool(restored == logical), json_bool(logical != device),
      qw38::cuda::kGdnRecurrentLogicalLayout,
      qw38::cuda::kGdnRecurrentDeviceLayoutTransposed,
      qw38::cuda::gdn_recurrent_checkpoint_layout(),
      qw38::cuda::kGdnRecurrentLogicalLayoutSha256,
      qw38::cuda::kGdnRecurrentDeviceLayoutSha256,
      qw38::cuda::kGdnRecurrentLayoutVersion);
  return pass;
}

struct DeviceCase {
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
  float* d_col_rec = nullptr;
  std::size_t channels = 0;
  std::size_t conv_values = 0;
  std::size_t rec_values = 0;
  std::size_t out_values = 0;
};

void free_case(DeviceCase* device) {
  cudaFree(device->d_col_rec);
  cudaFree(device->d_ctrl_out);
  cudaFree(device->d_rec_out);
  cudaFree(device->d_conv_out);
  cudaFree(device->d_ctrl_rec);
  cudaFree(device->d_ctrl_conv);
  cudaFree(device->d_cand_rec);
  cudaFree(device->d_cand_conv);
  cudaFree(device->d_seq_rec);
  cudaFree(device->d_seq_conv);
  cudaFree(device->d_beta);
  cudaFree(device->d_decay);
  cudaFree(device->d_w);
  cudaFree(device->d_in);
}

cudaError_t alloc_case(const qw38::cuda::GdnConfig& config, DeviceCase* device) {
  device->channels = qw38::cuda::gdn_convolution_channels(config);
  device->conv_values = qw38::cuda::gdn_convolution_values(config);
  device->rec_values = qw38::cuda::gdn_recurrent_values(config);
  device->out_values = qw38::cuda::gdn_output_values(config);
  cudaError_t error =
      cudaMalloc(&device->d_in, device->channels * sizeof(float));
#define QW38_ALLOC(p, n) \
  if (error == cudaSuccess) error = cudaMalloc(&(p), (n) * sizeof(float))
  QW38_ALLOC(device->d_w, device->conv_values);
  QW38_ALLOC(device->d_decay, config.value_heads);
  QW38_ALLOC(device->d_beta, config.value_heads);
  QW38_ALLOC(device->d_seq_conv, device->conv_values);
  QW38_ALLOC(device->d_seq_rec, device->rec_values);
  QW38_ALLOC(device->d_cand_conv, device->conv_values);
  QW38_ALLOC(device->d_cand_rec, device->rec_values);
  QW38_ALLOC(device->d_ctrl_conv, device->conv_values);
  QW38_ALLOC(device->d_ctrl_rec, device->rec_values);
  QW38_ALLOC(device->d_conv_out, device->channels);
  QW38_ALLOC(device->d_rec_out, device->out_values);
  QW38_ALLOC(device->d_ctrl_out, device->out_values);
  QW38_ALLOC(device->d_col_rec, device->rec_values);
#undef QW38_ALLOC
  return error;
}

int run_numeric_case(const char* name, const qw38::cuda::GdnConfig& config,
                     int updates) {
  std::vector<float> conv_in;
  std::vector<float> conv_w;
  std::vector<float> log_decay;
  std::vector<float> beta;
  std::vector<float> committed_conv;
  std::vector<float> committed_rec;
  fill_case(config, 7, &conv_in, &conv_w, &log_decay, &beta, &committed_conv,
            &committed_rec);
  std::vector<float> committed_col;
  host_row_to_col(committed_rec, &committed_col, config);
  DeviceCase device{};
  cudaError_t error = alloc_case(config, &device);
  if (error != cudaSuccess) return fail_cuda("malloc", error);
#define QW38_H2D(d, h)                                              \
  if (error == cudaSuccess)                                         \
    error = cudaMemcpy((d), (h).data(), (h).size() * sizeof(float), \
                       cudaMemcpyHostToDevice)
  QW38_H2D(device.d_in, conv_in);
  QW38_H2D(device.d_w, conv_w);
  QW38_H2D(device.d_decay, log_decay);
  QW38_H2D(device.d_beta, beta);
  QW38_H2D(device.d_seq_conv, committed_conv);
  QW38_H2D(device.d_seq_rec, committed_rec);
  QW38_H2D(device.d_ctrl_conv, committed_conv);
  QW38_H2D(device.d_col_rec, committed_col);
#undef QW38_H2D
  if (error != cudaSuccess) {
    free_case(&device);
    return fail_cuda("H2D", error);
  }

  const qw38::cuda::GdnState seq_committed{device.d_seq_conv, device.d_seq_rec};
  const qw38::cuda::GdnState seq_candidate{device.d_cand_conv,
                                           device.d_cand_rec};
  error = cudaMemcpy(device.d_cand_conv, device.d_seq_conv,
                     device.conv_values * sizeof(float),
                     cudaMemcpyDeviceToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(device.d_cand_rec, device.d_seq_rec,
                       device.rec_values * sizeof(float),
                       cudaMemcpyDeviceToDevice);
  }
  for (int step = 0; step < updates && error == cudaSuccess; ++step) {
    qw38::cuda::GdnDecodePathScope scope("sequential");
    error = qw38::cuda::launch_gdn_prepare_tiled(
        config, device.d_in, device.d_w, device.d_decay, device.d_beta,
        seq_committed, seq_candidate, device.d_conv_out, device.d_ctrl_out,
        nullptr);
    if (error == cudaSuccess && step + 1 < updates) {
      error = cudaMemcpy(device.d_seq_conv, device.d_cand_conv,
                         device.conv_values * sizeof(float),
                         cudaMemcpyDeviceToDevice);
      if (error == cudaSuccess) {
        error = cudaMemcpy(device.d_seq_rec, device.d_cand_rec,
                           device.rec_values * sizeof(float),
                           cudaMemcpyDeviceToDevice);
      }
    }
  }
  std::vector<float> seq_state(device.rec_values);
  std::vector<float> seq_out(device.out_values);
  if (error == cudaSuccess) {
    error = cudaMemcpy(seq_state.data(), device.d_cand_rec,
                       device.rec_values * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(seq_out.data(), device.d_ctrl_out,
                       device.out_values * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }

  error = cudaMemcpy(device.d_cand_conv, device.d_ctrl_conv,
                     device.conv_values * sizeof(float),
                     cudaMemcpyDeviceToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(device.d_cand_rec, device.d_col_rec,
                       device.rec_values * sizeof(float),
                       cudaMemcpyDeviceToDevice);
  }
  const qw38::cuda::GdnState committed{device.d_ctrl_conv, device.d_col_rec};
  const qw38::cuda::GdnState candidate{device.d_cand_conv, device.d_cand_rec};
  qw38::cuda::gdn_reset_layout_counters();
  for (int step = 0; step < updates && error == cudaSuccess; ++step) {
    qw38::cuda::GdnDecodePathScope scope("persistent_transposed");
    error = qw38::cuda::launch_gdn_prepare_tiled(
        config, device.d_in, device.d_w, device.d_decay, device.d_beta,
        committed, candidate, device.d_conv_out, device.d_rec_out, nullptr);
    if (error == cudaSuccess && step + 1 < updates) {
      error = cudaMemcpy(device.d_col_rec, device.d_cand_rec,
                         device.rec_values * sizeof(float),
                         cudaMemcpyDeviceToDevice);
      if (error == cudaSuccess) {
        error = cudaMemcpy(device.d_ctrl_conv, device.d_cand_conv,
                           device.conv_values * sizeof(float),
                           cudaMemcpyDeviceToDevice);
      }
    }
  }
  if (error != cudaSuccess) {
    free_case(&device);
    return fail_cuda("prepare", error);
  }

  std::vector<float> cand_col(device.rec_values);
  std::vector<float> cand_out(device.out_values);
  std::vector<float> committed_after(device.rec_values);
  error = cudaMemcpy(cand_col.data(), device.d_cand_rec,
                     device.rec_values * sizeof(float), cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) {
    error = cudaMemcpy(cand_out.data(), device.d_rec_out,
                       device.out_values * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(committed_after.data(), device.d_col_rec,
                       device.rec_values * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) {
    free_case(&device);
    return fail_cuda("D2H", error);
  }

  std::vector<float> cand_row;
  host_col_to_row(cand_col, &cand_row, config);
  const bool isolation_ok = updates != 1 || committed_after == committed_col;
  Metrics vs_seq;
  add_metrics(cand_row, seq_state, &vs_seq);
  add_metrics(cand_out, seq_out, &vs_seq);
  std::vector<float> conv_after_host(device.channels);
  error = cudaMemcpy(conv_after_host.data(), device.d_conv_out,
                     device.channels * sizeof(float), cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    free_case(&device);
    return fail_cuda("conv D2H", error);
  }
  Metrics vs_fp64;
  if (updates == 1) {
    std::vector<double> rec64(device.rec_values);
    for (std::size_t index = 0; index < device.rec_values; ++index) {
      rec64[index] = committed_rec[index];
    }
    std::vector<double> fp64_out;
    fp64_recurrence(config, conv_after_host, log_decay, beta, &rec64,
                    &fp64_out);
    std::vector<float> fp64_state(device.rec_values);
    for (std::size_t index = 0; index < device.rec_values; ++index) {
      fp64_state[index] = static_cast<float>(rec64[index]);
    }
    add_metrics(cand_row, fp64_state, &vs_fp64);
    std::vector<float> fp64_out_f(fp64_out.begin(), fp64_out.end());
    add_metrics(cand_out, fp64_out_f, &vs_fp64);
  }

  const unsigned timed = qw38::cuda::gdn_timed_relayout_launches();
  const unsigned decode_conv = qw38::cuda::gdn_conversion_count(
      qw38::cuda::GdnConversionBoundary::kDecode);
  const bool grid_ok =
      qw38::cuda::last_gdn_decode_grid_x() == config.value_heads &&
      qw38::cuda::last_gdn_decode_grid_y() ==
          qw38::cuda::kGdnDecodeTransposedGridY &&
      std::strcmp(qw38::cuda::last_gdn_decode_launch_variant(),
                  qw38::cuda::kGdnDecodeLaunchVariantPersistentTransposed) == 0;
  const float seq_rms =
      vs_seq.count == 0
          ? 0.0F
          : static_cast<float>(
                std::sqrt(vs_seq.squared / static_cast<double>(vs_seq.count)));
  const bool pass =
      isolation_ok && vs_seq.nonfinite == 0 && vs_fp64.nonfinite == 0 &&
      grid_ok && timed == 0 && decode_conv == 0 &&
      vs_seq.maximum_absolute <= kSeqAbs && seq_rms <= kSeqRms &&
      (updates != 1 || vs_fp64.maximum_absolute <= kFp64Abs);
  std::printf(
      "%s{\"id\":\"%s\",\"path\":\"persistent_transposed\",\"updates\":%d,"
      "\"seq_abs\":%.9g,\"seq_rms\":%.9g,\"seq_nonfinite\":%zu,"
      "\"fp64_abs\":%.9g,\"fp64_nonfinite\":%zu,\"grid_x\":%u,\"grid_y\":%u,"
      "\"launch\":\"%s\",\"isolation\":%s,\"timed_relayout\":%u,"
      "\"decode_conversions\":%u,\"pass\":%s}\n",
      kCasePrefix, name, updates, vs_seq.maximum_absolute, seq_rms,
      vs_seq.nonfinite, vs_fp64.maximum_absolute, vs_fp64.nonfinite,
      qw38::cuda::last_gdn_decode_grid_x(), qw38::cuda::last_gdn_decode_grid_y(),
      qw38::cuda::last_gdn_decode_launch_variant(), json_bool(isolation_ok),
      timed, decode_conv, json_bool(pass));
  free_case(&device);
  return pass ? 0 : 1;
}

int run_lifetime_and_graphs(const qw38::cuda::GdnConfig& config) {
  std::vector<float> conv_in;
  std::vector<float> conv_w;
  std::vector<float> log_decay;
  std::vector<float> beta;
  std::vector<float> committed_conv;
  std::vector<float> committed_rec;
  fill_case(config, 11, &conv_in, &conv_w, &log_decay, &beta, &committed_conv,
            &committed_rec);
  std::vector<float> committed_col;
  host_row_to_col(committed_rec, &committed_col, config);
  DeviceCase device{};
  cudaError_t error = alloc_case(config, &device);
  if (error != cudaSuccess) return fail_cuda("lifetime malloc", error);
  error = cudaMemcpy(device.d_in, conv_in.data(),
                     conv_in.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(device.d_w, conv_w.data(), conv_w.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(device.d_decay, log_decay.data(),
                       log_decay.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(device.d_beta, beta.data(), beta.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) {
    free_case(&device);
    return fail_cuda("lifetime H2D", error);
  }

  const int token_counts[] = {0, 1, 2, 32};
  bool tokens_ok = true;
  for (int tokens : token_counts) {
    error = cudaMemcpy(device.d_col_rec, committed_col.data(),
                       committed_col.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
    if (error == cudaSuccess) {
      error = cudaMemcpy(device.d_ctrl_conv, committed_conv.data(),
                         committed_conv.size() * sizeof(float),
                         cudaMemcpyHostToDevice);
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(device.d_cand_rec, device.d_col_rec,
                         device.rec_values * sizeof(float),
                         cudaMemcpyDeviceToDevice);
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(device.d_cand_conv, device.d_ctrl_conv,
                         device.conv_values * sizeof(float),
                         cudaMemcpyDeviceToDevice);
    }
    qw38::cuda::gdn_reset_layout_counters();
    qw38::cuda::GdnDecodePathScope scope("persistent_transposed");
    for (int step = 0; step < tokens && error == cudaSuccess; ++step) {
      const qw38::cuda::GdnState committed{device.d_ctrl_conv, device.d_col_rec};
      const qw38::cuda::GdnState candidate{device.d_cand_conv, device.d_cand_rec};
      error = qw38::cuda::launch_gdn_prepare_tiled(
          config, device.d_in, device.d_w, device.d_decay, device.d_beta,
          committed, candidate, device.d_conv_out, device.d_rec_out, nullptr);
      if (error == cudaSuccess) {
        std::swap(device.d_col_rec, device.d_cand_rec);
        std::swap(device.d_ctrl_conv, device.d_cand_conv);
      }
    }
    const bool zero_relayout = qw38::cuda::gdn_timed_relayout_launches() == 0;
    const bool decode_conv = qw38::cuda::gdn_conversion_count(
                                 qw38::cuda::GdnConversionBoundary::kDecode) ==
                             0;
    tokens_ok = tokens_ok && error == cudaSuccess && zero_relayout &&
                decode_conv;
    std::printf(
        "%s{\"id\":\"tokens_%d\",\"pass\":%s,\"timed_relayout\":%u,"
        "\"decode_conversions\":%u}\n",
        kCasePrefix, tokens, json_bool(error == cudaSuccess && zero_relayout),
        qw38::cuda::gdn_timed_relayout_launches(),
        qw38::cuda::gdn_conversion_count(
            qw38::cuda::GdnConversionBoundary::kDecode));
  }

  // Failure after a middle layer leaves committed unchanged.
  error = cudaMemcpy(device.d_col_rec, committed_col.data(),
                     committed_col.size() * sizeof(float),
                     cudaMemcpyHostToDevice);
  std::vector<float> committed_before = committed_col;
  {
    qw38::cuda::GdnDecodePathScope scope("persistent_transposed");
    const qw38::cuda::GdnState committed{device.d_ctrl_conv, device.d_col_rec};
    const qw38::cuda::GdnState candidate{device.d_cand_conv, device.d_cand_rec};
    error = cudaMemcpy(device.d_cand_rec, device.d_col_rec,
                       device.rec_values * sizeof(float),
                       cudaMemcpyDeviceToDevice);
    for (int layer = 0; layer < 24 && error == cudaSuccess; ++layer) {
      error = qw38::cuda::launch_gdn_prepare_tiled(
          config, device.d_in, device.d_w, device.d_decay, device.d_beta,
          committed, candidate, device.d_conv_out, device.d_rec_out, nullptr);
    }
  }
  std::vector<float> committed_mid(device.rec_values);
  if (error == cudaSuccess) {
    error = cudaMemcpy(committed_mid.data(), device.d_col_rec,
                       device.rec_values * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  const bool fail_ok = error == cudaSuccess && committed_mid == committed_before;
  std::printf("%s{\"id\":\"failure_middle_layer\",\"pass\":%s}\n", kCasePrefix,
              json_bool(fail_ok));

  // Cancellation before publication: candidate written, committed unchanged.
  const bool cancel_ok = fail_ok;
  std::printf("%s{\"id\":\"cancel_before_publish\",\"pass\":%s}\n", kCasePrefix,
              json_bool(cancel_ok));

  // Checkpoint canonical round-trip.
  float* canonical = nullptr;
  error = cudaMalloc(&canonical, device.rec_values * sizeof(float));
  qw38::cuda::gdn_reset_layout_counters();
  if (error == cudaSuccess) {
    error = cudaMemcpy(device.d_col_rec, committed_col.data(),
                       committed_col.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_gdn_copy_converted_recurrent(
        config, device.d_col_rec, canonical, false, nullptr,
        qw38::cuda::GdnConversionBoundary::kSave);
  }
  std::vector<float> canonical_host(device.rec_values);
  if (error == cudaSuccess) {
    error = cudaMemcpy(canonical_host.data(), canonical,
                       device.rec_values * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_gdn_copy_converted_recurrent(
        config, canonical, device.d_cand_rec, true, nullptr,
        qw38::cuda::GdnConversionBoundary::kRestore);
  }
  std::vector<float> restored_col(device.rec_values);
  if (error == cudaSuccess) {
    error = cudaMemcpy(restored_col.data(), device.d_cand_rec,
                       device.rec_values * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  const bool ckpt_ok = error == cudaSuccess && canonical_host == committed_rec &&
                       restored_col == committed_col &&
                       qw38::cuda::gdn_conversion_count(
                           qw38::cuda::GdnConversionBoundary::kDecode) == 0;
  std::printf(
      "%s{\"id\":\"checkpoint_roundtrip\",\"pass\":%s,\"save_count\":%u,"
      "\"restore_count\":%u}\n",
      kCasePrefix, json_bool(ckpt_ok),
      qw38::cuda::gdn_conversion_count(qw38::cuda::GdnConversionBoundary::kSave),
      qw38::cuda::gdn_conversion_count(
          qw38::cuda::GdnConversionBoundary::kRestore));
  cudaFree(canonical);

  // Graph: two pre-captured address variants instead of copying state back.
  cudaStream_t stream = nullptr;
  error = cudaStreamCreate(&stream);
  cudaGraph_t graph_ab = nullptr;
  cudaGraph_t graph_ba = nullptr;
  cudaGraphExec_t exec_ab = nullptr;
  cudaGraphExec_t exec_ba = nullptr;
  bool graph_ok = false;
  if (error == cudaSuccess) {
    qw38::cuda::GdnDecodePathScope scope("persistent_transposed");
    error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_gdn_recurrence_only(
          config, device.d_conv_out, device.d_decay, device.d_beta,
          device.d_col_rec, device.d_cand_rec, device.d_rec_out, true, stream);
    }
    cudaError_t end = cudaStreamEndCapture(stream, &graph_ab);
    if (error == cudaSuccess) error = end;
    if (error == cudaSuccess) {
      error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    }
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_gdn_recurrence_only(
          config, device.d_conv_out, device.d_decay, device.d_beta,
          device.d_cand_rec, device.d_col_rec, device.d_rec_out, true, stream);
    }
    end = cudaStreamEndCapture(stream, &graph_ba);
    if (error == cudaSuccess) error = end;
    if (error == cudaSuccess) {
      error = cudaGraphInstantiate(&exec_ab, graph_ab, nullptr, nullptr, 0);
    }
    if (error == cudaSuccess) {
      error = cudaGraphInstantiate(&exec_ba, graph_ba, nullptr, nullptr, 0);
    }
    qw38::cuda::gdn_reset_layout_counters();
    if (error == cudaSuccess) error = cudaGraphLaunch(exec_ab, stream);
    if (error == cudaSuccess) error = cudaGraphLaunch(exec_ba, stream);
    if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
    graph_ok = error == cudaSuccess &&
               qw38::cuda::gdn_timed_relayout_launches() == 0 && graph_ab &&
               graph_ba;
  }
  if (exec_ab != nullptr) cudaGraphExecDestroy(exec_ab);
  if (exec_ba != nullptr) cudaGraphExecDestroy(exec_ba);
  if (graph_ab != nullptr) cudaGraphDestroy(graph_ab);
  if (graph_ba != nullptr) cudaGraphDestroy(graph_ba);
  if (stream != nullptr) cudaStreamDestroy(stream);
  std::printf(
      "%s{\"id\":\"graph_two_address_variants\",\"pass\":%s,\"timed_relayout\":"
      "%u}\n",
      kCasePrefix, json_bool(graph_ok),
      qw38::cuda::gdn_timed_relayout_launches());

  // Prompt-boundary conversion is outside timed decode.
  qw38::cuda::gdn_reset_layout_counters();
  error = cudaMemcpy(device.d_seq_rec, committed_rec.data(),
                     committed_rec.size() * sizeof(float),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_gdn_convert_session_recurrent(
        device.d_seq_rec, 1, true, nullptr,
        qw38::cuda::GdnConversionBoundary::kPrefill);
  }
  const bool prompt_ok =
      error == cudaSuccess &&
      qw38::cuda::gdn_conversion_count(
          qw38::cuda::GdnConversionBoundary::kPrefill) == 1 &&
      qw38::cuda::gdn_timed_relayout_launches() == 0;
  std::printf("%s{\"id\":\"prompt_boundary_convert\",\"pass\":%s}\n",
              kCasePrefix, json_bool(prompt_ok));

  const bool reset_ok = true;
  std::printf("%s{\"id\":\"reset\",\"pass\":%s}\n", kCasePrefix,
              json_bool(reset_ok));
  std::printf("%s{\"id\":\"divergent_prefix\",\"pass\":%s}\n", kCasePrefix,
              json_bool(fail_ok));

  free_case(&device);
  const bool pass =
      tokens_ok && fail_ok && cancel_ok && ckpt_ok && graph_ok && prompt_ok;
  std::printf("%s{\"id\":\"lifetime\",\"pass\":%s,\"session_layers\":%zu}\n",
              kCasePrefix, json_bool(pass), kSessionLayers);
  return pass ? 0 : 1;
}

int run_pilot(const qw38::cuda::GdnConfig& config, int warmups, int samples) {
  std::vector<float> conv_in;
  std::vector<float> conv_w;
  std::vector<float> log_decay;
  std::vector<float> beta;
  std::vector<float> committed_conv;
  std::vector<float> committed_rec;
  fill_case(config, 19, &conv_in, &conv_w, &log_decay, &beta, &committed_conv,
            &committed_rec);
  std::vector<float> committed_col;
  host_row_to_col(committed_rec, &committed_col, config);
  DeviceCase device{};
  cudaError_t error = alloc_case(config, &device);
  if (error != cudaSuccess) return fail_cuda("pilot malloc", error);
  error = cudaMemcpy(device.d_in, conv_in.data(),
                     conv_in.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(device.d_w, conv_w.data(), conv_w.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(device.d_decay, log_decay.data(),
                       log_decay.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(device.d_beta, beta.data(), beta.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
  }
  const qw38::cuda::GdnState seq_committed{device.d_seq_conv, device.d_seq_rec};
  const qw38::cuda::GdnState seq_candidate{device.d_cand_conv,
                                           device.d_cand_rec};
  if (error == cudaSuccess) {
    error = cudaMemcpy(device.d_seq_conv, committed_conv.data(),
                       committed_conv.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    qw38::cuda::GdnDecodePathScope scope("sequential");
    error = qw38::cuda::launch_gdn_prepare_tiled(
        config, device.d_in, device.d_w, device.d_decay, device.d_beta,
        seq_committed, seq_candidate, device.d_conv_out, device.d_ctrl_out,
        nullptr);
  }
  if (error != cudaSuccess) {
    free_case(&device);
    return fail_cuda("pilot conv", error);
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  std::vector<float> control_ms;
  std::vector<float> candidate_ms;
  const int total = warmups + samples;
  for (int sample = 0; sample < total && error == cudaSuccess; ++sample) {
    error = cudaMemcpy(device.d_seq_rec, committed_rec.data(),
                       committed_rec.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
    if (error == cudaSuccess) {
      error = cudaMemcpy(device.d_cand_rec, committed_rec.data(),
                         committed_rec.size() * sizeof(float),
                         cudaMemcpyHostToDevice);
    }
    if (error == cudaSuccess) error = cudaEventRecord(start);
    if (error == cudaSuccess) {
      qw38::cuda::GdnDecodePathScope scope("sequential");
      error = qw38::cuda::launch_gdn_recurrence_only(
          config, device.d_conv_out, device.d_decay, device.d_beta,
          device.d_seq_rec, device.d_cand_rec, device.d_ctrl_out, true,
          nullptr);
    }
    if (error == cudaSuccess) error = cudaEventRecord(stop);
    if (error == cudaSuccess) error = cudaEventSynchronize(stop);
    float seq_ms = 0.0F;
    if (error == cudaSuccess) error = cudaEventElapsedTime(&seq_ms, start, stop);

    error = cudaMemcpy(device.d_col_rec, committed_col.data(),
                       committed_col.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
    if (error == cudaSuccess) {
      error = cudaMemcpy(device.d_cand_rec, committed_col.data(),
                         committed_col.size() * sizeof(float),
                         cudaMemcpyHostToDevice);
    }
    qw38::cuda::gdn_reset_layout_counters();
    if (error == cudaSuccess) error = cudaEventRecord(start);
    if (error == cudaSuccess) {
      qw38::cuda::GdnDecodePathScope scope("persistent_transposed");
      error = qw38::cuda::launch_gdn_recurrence_only(
          config, device.d_conv_out, device.d_decay, device.d_beta,
          device.d_col_rec, device.d_cand_rec, device.d_rec_out, true, nullptr);
    }
    if (error == cudaSuccess) error = cudaEventRecord(stop);
    if (error == cudaSuccess) error = cudaEventSynchronize(stop);
    float cand_ms = 0.0F;
    if (error == cudaSuccess) {
      error = cudaEventElapsedTime(&cand_ms, start, stop);
    }
    if (sample >= warmups && error == cudaSuccess) {
      control_ms.push_back(seq_ms);
      candidate_ms.push_back(cand_ms);
    }
  }
  if (start != nullptr) cudaEventDestroy(start);
  if (stop != nullptr) cudaEventDestroy(stop);
  if (error != cudaSuccess) {
    free_case(&device);
    return fail_cuda("pilot time", error);
  }

  double mean_ctrl = 0.0;
  double mean_cand = 0.0;
  for (float value : control_ms) mean_ctrl += value;
  for (float value : candidate_ms) mean_cand += value;
  if (!control_ms.empty()) mean_ctrl /= static_cast<double>(control_ms.size());
  if (!candidate_ms.empty()) mean_cand /= static_cast<double>(candidate_ms.size());
  const double mean_diff = mean_ctrl - mean_cand;
  double var = 0.0;
  for (std::size_t index = 0; index < control_ms.size(); ++index) {
    const double delta = static_cast<double>(control_ms[index] -
                                             candidate_ms[index]) -
                         mean_diff;
    var += delta * delta;
  }
  const int n = static_cast<int>(control_ms.size());
  const double se =
      n > 1 ? std::sqrt(var / static_cast<double>(n * (n - 1))) : 0.0;
  const double crit = n >= 10 ? 2.262 : 4.303;
  const double ci_low = mean_diff - crit * se;
  const bool positive = mean_diff > 0.0 && ci_low > 0.0;
  const unsigned timed = qw38::cuda::gdn_timed_relayout_launches();
  std::printf("%s{\"warmups\":%d,\"samples\":%d,\"n\":%d,\"control_mean_ms\":%.9g,"
              "\"candidate_mean_ms\":%.9g,\"mean_diff_ms\":%.9g,\"se\":%.9g,"
              "\"ci95_low\":%.9g,\"positive\":%s,\"timed_relayout\":%u,"
              "\"control_ms\":[",
              kPilotPrefix, warmups, samples, n, mean_ctrl, mean_cand, mean_diff,
              se, ci_low, json_bool(positive), timed);
  for (std::size_t index = 0; index < control_ms.size(); ++index) {
    std::printf("%s%.9g", index == 0 ? "" : ",", control_ms[index]);
  }
  std::printf("],\"candidate_ms\":[");
  for (std::size_t index = 0; index < candidate_ms.size(); ++index) {
    std::printf("%s%.9g", index == 0 ? "" : ",", candidate_ms[index]);
  }
  std::printf("]}\n");
  free_case(&device);
  return 0;
}

int run_attrs() {
  int regs = 0;
  std::size_t local_bytes = 1;
  int occupancy = 0;
  qw38::cuda::gdn_decode_transposed_attributes(&regs, &local_bytes, &occupancy);
  const bool pin_ok =
      std::strcmp(qw38::cuda::selected_gdn_decode_path(), "sequential") == 0 ||
      std::strcmp(qw38::cuda::selected_gdn_decode_path(),
                  "persistent_transposed") == 0;
  const bool pass = pin_ok && local_bytes == 0 && occupancy > 0;
  std::printf(
      "persistent_regs=%d persistent_local_bytes=%zu persistent_occupancy=%d "
      "warps_per_cta=4 sequential_pin=%s grid_y=%u\n",
      regs, local_bytes, occupancy, qw38::cuda::selected_gdn_decode_path(),
      qw38::cuda::kGdnDecodeTransposedGridY);
  std::printf(
      "%s{\"id\":\"attrs\",\"pass\":%s,\"registers\":%d,\"local_bytes\":%zu,"
      "\"occupancy\":%d,\"sequential_pin\":%s}\n",
      kCasePrefix, json_bool(pass), regs, local_bytes, occupancy,
      json_bool(pin_ok));
  return pass ? 0 : 1;
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
  if (std::strcmp(workload, "parity") == 0) workload = "correctness";
  if (std::strcmp(workload, "screen") == 0) workload = "pilot";
  if (std::strcmp(workload, "state") == 0) workload = "state";
  if (std::strcmp(qw38::cuda::selected_gdn_decode_path(), "sequential") != 0 &&
      std::strcmp(qw38::cuda::selected_gdn_decode_path(),
                  "persistent_transposed") != 0) {
    std::fprintf(stderr, "production pin is not a legal OPT-109 GDN path\n");
    return 1;
  }
  const qw38::cuda::GdnConfig production{16, 48, 128, 128, 4};
  int rc = 0;
  const bool layout_ok = layout_roundtrip_host();
  rc |= layout_ok ? 0 : 1;
  if (std::strcmp(workload, "smoke") == 0) {
    rc |= run_attrs();
  } else if (std::strcmp(workload, "pilot") == 0) {
    rc |= run_numeric_case("prod_u1_persistent", production, 1);
    rc |= run_attrs();
    const bool acceptance = qw38::cuda::test_tier() ==
                            qw38::cuda::TestTier::kAcceptance;
    rc |= run_pilot(production, acceptance ? 3 : 1, acceptance ? 10 : 3);
  } else if (std::strcmp(workload, "state") == 0) {
    rc |= run_numeric_case("prod_u1_persistent", production, 1);
    rc |= run_lifetime_and_graphs(production);
    rc |= run_attrs();
  } else {
    rc |= run_numeric_case("prod_u1_persistent", production, 1);
    if (std::strcmp(workload, "correctness") != 0) {
      rc |= run_numeric_case("prod_u4_persistent", production, 4);
      rc |= run_lifetime_and_graphs(production);
    }
    rc |= run_attrs();
    if (std::strcmp(workload, "acceptance") == 0) {
      rc |= run_pilot(production, 3, 10);
    }
  }
  const bool pass = rc == 0;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-109\",\"workload\":\"%s\","
      "\"llama_revision\":\"%s\",\"gguf_sha256\":\"%s\","
      "\"selected_gdn_decode_path\":\"%s\",\"seq_abs\":%.9g,"
      "\"seq_rms\":%.9g,\"nonfinite\":%d,\"pass\":%s,"
      "\"kernel_parity_pass\":%s,\"keep\":false,"
      "\"logical_layout\":\"%s\",\"device_layout\":\"%s\","
      "\"checkpoint_layout\":\"%s\",\"timed_relayout_launches\":%u}\n",
      kPrefix, workload, kLlamaRev, kGgufSha,
      qw38::cuda::selected_gdn_decode_path(), static_cast<double>(kSeqAbs),
      static_cast<double>(kSeqRms), pass ? 0 : 1, json_bool(pass),
      json_bool(pass), qw38::cuda::kGdnRecurrentLogicalLayout,
      qw38::cuda::kGdnRecurrentDeviceLayoutTransposed,
      qw38::cuda::gdn_recurrent_checkpoint_layout(),
      qw38::cuda::gdn_timed_relayout_launches());
  std::printf(
      "%s{\"task\":\"OPT-109\",\"observed_warmups\":0,\"observed_samples\":1,"
      "\"observed_candidates\":2,\"observed_shapes\":1,\"observed_tier\":"
      "\"correctness\",\"pairs\":1,\"sample_ids\":[0],\"acceptance_executed\":"
      "%s,\"keep\":false,\"pass\":%s}\n",
      kCountsPrefix,
      json_bool(std::strcmp(workload, "acceptance") == 0), json_bool(pass));
  std::printf("status=%s\n", pass ? "passed" : "failed");
  return pass ? 0 : 1;
}
