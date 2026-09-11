#include "full_scheduler.h"
#include "mixer.h"
#include "model.h"
#include "q8_decode_path.cuh"
#include "quant.h"
#include "quant_mmv.h"
#include "test_tier.h"
#include "weights.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr char kPrefix[] = "QW38_OPT064_Q8_ROWS_RESULT=";
constexpr std::size_t kQ80Bytes = 34;
constexpr std::size_t kQ80Values = 32;
constexpr std::size_t kGuard = 64;
constexpr std::size_t kGdnLayers = 48;
constexpr std::size_t kAttnLayers = 16;
constexpr const char* kGgufSha =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr const char* kLlamaRev =
    "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr float kStagedAbs = 3.0e-4F;
constexpr float kLayoutAbs = 1.0e-5F;
constexpr int kSampledRows = 16;
constexpr int kCaptures = 4;

float abs_gate(std::size_t columns) {
  const float scaled =
      kStagedAbs *
      std::sqrt(std::max(1.0F, static_cast<float>(columns) / 32.0F));
  return std::max(kStagedAbs, scaled);
}

struct Layout {
  const char* id;
  unsigned int rows_per_cta;
  unsigned int warps_per_row;
};

constexpr Layout kLayouts[] = {
    {"r1_w4", 1, 4},
    {"r2_w2", 2, 2},
    {"r4_w1", 4, 1},
    {"r8_w1", 8, 1},
};
constexpr int kLayoutCount = 4;

struct Options final {
  const char* workload = nullptr;
  const char* model = nullptr;
};

const char* json_bool(bool value) { return value ? "true" : "false"; }

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

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s\n", status.message().c_str());
  return 1;
}

void write_u16(std::uint8_t* bytes, std::uint16_t value) {
  bytes[0] = static_cast<std::uint8_t>(value & 0xFFU);
  bytes[1] = static_cast<std::uint8_t>(value >> 8U);
}

float unit_normal(std::uint32_t index, std::uint32_t seed) {
  std::uint32_t x = index * 747796405U + seed;
  x ^= x >> 16U;
  x *= 0x7feb352dU;
  x ^= x >> 15U;
  x *= 0x846ca68bU;
  x ^= x >> 16U;
  const float u1 =
      (static_cast<float>(x & 0x00FFFFFFU) + 1.0F) / 16777217.0F;
  x = x * 1664525U + 1013904223U;
  const float u2 = static_cast<float>(x & 0x00FFFFFFU) / 16777216.0F;
  return std::sqrt(-2.0F * std::log(u1)) *
         std::cos(6.283185307179586F * u2);
}

std::uint16_t float_to_bf16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t rounding_bias = ((bits >> 16U) & 1U) + 0x7FFFU;
  return static_cast<std::uint16_t>((bits + rounding_bias) >> 16U);
}

__nv_bfloat16 from_bf16_bits(std::uint16_t bits) {
  __nv_bfloat16_raw raw;
  raw.x = bits;
  return __nv_bfloat16(raw);
}

void fill_weights(std::size_t rows, std::size_t columns, const char* pattern,
                  std::vector<std::uint8_t>* weights) {
  weights->assign(rows * (columns / kQ80Values) * kQ80Bytes, 0);
  for (std::size_t index = 0; index < weights->size(); ++index) {
    (*weights)[index] = static_cast<std::uint8_t>((index * 73 + 19) & 0xFFU);
  }
  for (std::size_t offset = 0; offset < weights->size(); offset += kQ80Bytes) {
    write_u16(weights->data() + offset, 0x3C00U);
    if (std::strcmp(pattern, "extremes") == 0) {
      for (int i = 0; i < 32; ++i) {
        weights->data()[offset + 2 + static_cast<std::size_t>(i)] =
            (i % 2 == 0) ? 0x80U : 0x7FU;
      }
    }
  }
}

void fill_activation(const char* pattern, std::size_t columns, std::uint32_t seed,
                     std::vector<__nv_bfloat16>* activation) {
  activation->resize(columns);
  for (std::size_t column = 0; column < columns; ++column) {
    float value = unit_normal(static_cast<std::uint32_t>(column), seed);
    if (std::strcmp(pattern, "zero") == 0) {
      value = 0.0F;
    } else if (std::strcmp(pattern, "cancel") == 0) {
      value = (column % 2U == 0) ? 8.0F : -8.0F;
    } else if (std::strcmp(pattern, "extremes") == 0) {
      value = (column % 2U == 0) ? 127.0F : -127.0F;
    }
    (*activation)[column] = from_bf16_bits(float_to_bf16(value));
  }
}

void host_from_gpu_q8_1(const std::uint8_t* bytes, std::size_t columns,
                        std::vector<double>* staged) {
  staged->assign(columns, 0.0);
  const std::size_t groups = columns / kQ80Values;
  for (std::size_t group = 0; group < groups; ++group) {
    const std::uint8_t* block = bytes + group * 36U;
    __half scale_bits;
    std::memcpy(&scale_bits, block, sizeof(__half));
    const double scale = static_cast<double>(__half2float(scale_bits));
    for (int i = 0; i < 32; ++i) {
      const std::int8_t q = static_cast<std::int8_t>(block[4 + i]);
      (*staged)[group * 32U + static_cast<std::size_t>(i)] =
          scale * static_cast<double>(q);
    }
  }
}

bool host_fp64(const std::vector<std::uint8_t>& weights, std::size_t rows,
               std::size_t columns, const std::vector<double>& activation,
               const std::vector<std::size_t>& sample,
               std::vector<double>* output) {
  output->assign(sample.size(), 0.0);
  std::vector<float> decoded(kQ80Values);
  for (std::size_t s = 0; s < sample.size(); ++s) {
    const std::size_t row = sample[s];
    if (row >= rows) {
      (*output)[s] = std::nan("");
      continue;
    }
    double sum = 0.0;
    for (std::size_t block = 0; block < columns / kQ80Values; ++block) {
      const std::uint8_t* packed =
          weights.data() +
          (row * (columns / kQ80Values) + block) * kQ80Bytes;
      if (!qw38::internal::decode_q8_0(packed, kQ80Bytes, decoded.data(),
                                       decoded.size())
               .is_ok()) {
        return false;
      }
      for (std::size_t within = 0; within < kQ80Values; ++within) {
        sum += static_cast<double>(decoded[within]) *
               activation[block * kQ80Values + within];
      }
    }
    (*output)[s] = sum;
  }
  return true;
}

std::vector<std::size_t> all_rows(std::size_t rows) {
  std::vector<std::size_t> sample(rows);
  for (std::size_t row = 0; row < rows; ++row) sample[row] = row;
  return sample;
}

std::vector<std::size_t> sample_rows(std::size_t rows) {
  std::vector<std::size_t> sample;
  const std::size_t take = std::min(rows, static_cast<std::size_t>(kSampledRows));
  for (std::size_t i = 0; i < take; ++i) {
    sample.push_back((i * rows) / take);
  }
  return sample;
}

bool guards_ok(const std::vector<std::uint8_t>& bytes, std::size_t inner) {
  for (std::size_t i = 0; i < kGuard; ++i) {
    if (bytes[i] != 0xA5U || bytes[kGuard + inner + i] != 0x5AU) return false;
  }
  return true;
}

int compare_rows(const std::vector<float>& got, const std::vector<double>& ref,
                 const std::vector<std::size_t>& sample, float abs_gate,
                 const char* label) {
  float max_abs = 0.0F;
  int nonfinite = 0;
  for (std::size_t i = 0; i < sample.size(); ++i) {
    const float value = got[sample[i]];
    const double expect = ref[i];
    if (!std::isfinite(value) || !std::isfinite(expect)) {
      ++nonfinite;
      continue;
    }
    max_abs = std::max(max_abs,
                       std::fabs(value - static_cast<float>(expect)));
  }
  std::printf("compare label=%s max_abs=%.9g nonfinite=%d gate=%.9g\n", label,
              static_cast<double>(max_abs), nonfinite,
              static_cast<double>(abs_gate));
  return nonfinite == 0 && max_abs <= abs_gate ? 0 : 1;
}

cudaError_t launch_layout(const Layout& layout, const std::uint8_t* weights,
                          std::size_t rows, std::size_t columns,
                          const void* staged, float* output) {
  return qw38::cuda::launch_q8_coop_mmv_prequant(
      weights, rows, columns, staged, output, layout.rows_per_cta,
      layout.warps_per_row, nullptr);
}

struct RunCase {
  std::vector<std::uint8_t> host_w;
  std::vector<__nv_bfloat16> host_a;
  std::uint8_t* d_w = nullptr;
  __nv_bfloat16* d_a = nullptr;
  void* d_s = nullptr;
  float* d_o = nullptr;
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::size_t inner_bytes = 0;
};

std::size_t q81_bytes(std::size_t columns) {
  return (columns / kQ80Values) * 36U;
}

int alloc_case(std::size_t rows, std::size_t cols, RunCase* box) {
  box->rows = rows;
  box->cols = cols;
  box->inner_bytes = rows * (cols / kQ80Values) * kQ80Bytes;
  const std::size_t guarded = box->inner_bytes + 2 * kGuard;
  cudaError_t error = cudaMalloc(&box->d_w, guarded);
  if (error == cudaSuccess) {
    error = cudaMalloc(&box->d_a, cols * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&box->d_s, q81_bytes(cols));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&box->d_o, (rows + 8) * sizeof(float));
  }
  if (error != cudaSuccess) return fail_cuda("alloc_case", error);
  return 0;
}

void free_case(RunCase* box) {
  cudaFree(box->d_w);
  cudaFree(box->d_a);
  cudaFree(box->d_s);
  cudaFree(box->d_o);
  box->d_w = nullptr;
  box->d_a = nullptr;
  box->d_s = nullptr;
  box->d_o = nullptr;
}

int upload_case(RunCase* box) {
  std::vector<std::uint8_t> guarded(box->inner_bytes + 2 * kGuard, 0);
  std::fill(guarded.begin(), guarded.begin() + static_cast<std::ptrdiff_t>(kGuard),
            static_cast<std::uint8_t>(0xA5U));
  std::memcpy(guarded.data() + kGuard, box->host_w.data(), box->inner_bytes);
  std::fill(guarded.begin() + static_cast<std::ptrdiff_t>(kGuard + box->inner_bytes),
            guarded.end(), static_cast<std::uint8_t>(0x5AU));
  cudaError_t error =
      cudaMemcpy(box->d_w, guarded.data(), guarded.size(), cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(box->d_a, box->host_a.data(),
                       box->cols * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) return fail_cuda("upload_case", error);
  return 0;
}

int check_guards(RunCase* box, const char* label) {
  std::vector<std::uint8_t> after(box->inner_bytes + 2 * kGuard, 0);
  const cudaError_t error =
      cudaMemcpy(after.data(), box->d_w, after.size(), cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return fail_cuda("guard copy", error);
  if (!guards_ok(after, box->inner_bytes)) {
    std::fprintf(stderr, "tail guard corruption %s\n", label);
    return 1;
  }
  return 0;
}

int run_layouts_vs_ref(const char* name, std::size_t rows, std::size_t cols,
                       const char* wpattern, const char* apattern,
                       std::uint32_t seed, bool full_rows, float staged_gate,
                       bool complete_ref) {
  staged_gate = std::max(staged_gate, abs_gate(cols));
  RunCase box;
  if (alloc_case(rows, cols, &box) != 0) return 1;
  fill_weights(rows, cols, wpattern, &box.host_w);
  fill_activation(apattern, cols, seed, &box.host_a);
  if (upload_case(&box) != 0) {
    free_case(&box);
    return 1;
  }
  cudaError_t error =
      qw38::cuda::launch_quantize_bf16_q8_1(box.d_a, box.d_s, cols, nullptr);
  if (error != cudaSuccess) {
    free_case(&box);
    return fail_cuda("quantize", error);
  }
  std::vector<std::uint8_t> staged_bytes(q81_bytes(cols), 0);
  error = cudaMemcpy(staged_bytes.data(), box.d_s, staged_bytes.size(),
                     cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    free_case(&box);
    return fail_cuda("staged d2h", error);
  }
  std::vector<double> orig(cols);
  for (std::size_t i = 0; i < cols; ++i) {
    orig[i] = static_cast<double>(__bfloat162float(box.host_a[i]));
  }
  std::vector<double> staged;
  host_from_gpu_q8_1(staged_bytes.data(), cols, &staged);
  const std::vector<std::size_t> sample =
      full_rows ? all_rows(rows) : sample_rows(rows);
  std::vector<double> ref_staged;
  std::vector<double> ref_orig;
  if (!host_fp64(box.host_w, rows, cols, staged, sample, &ref_staged) ||
      !host_fp64(box.host_w, rows, cols, orig, sample, &ref_orig)) {
    free_case(&box);
    std::fprintf(stderr, "independent decode failed %s\n", name);
    return 1;
  }
  int rc = 0;
  std::vector<float> control;
  for (int layout = 0; layout < kLayoutCount; ++layout) {
    error = cudaMemset(box.d_o, 0xFF, (rows + 8) * sizeof(float));
    if (error == cudaSuccess) {
      error = launch_layout(kLayouts[layout], box.d_w + kGuard, rows, cols,
                            box.d_s, box.d_o);
    }
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
      free_case(&box);
      return fail_cuda(kLayouts[layout].id, error);
    }
    std::vector<float> host(rows + 8, 0.0F);
    error = cudaMemcpy(host.data(), box.d_o, host.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
      free_case(&box);
      return fail_cuda("d2h", error);
    }
    char label[96];
    std::snprintf(label, sizeof(label), "%s_%s_staged", name,
                  kLayouts[layout].id);
    rc |= compare_rows(host, ref_staged, sample, staged_gate, label);
    if (complete_ref) {
      std::snprintf(label, sizeof(label), "%s_%s_complete", name,
                    kLayouts[layout].id);
      rc |= compare_rows(host, ref_orig, sample, 80.0F, label);
    }
    if (layout == 0) {
      control = host;
    } else {
      float max_abs = 0.0F;
      for (std::size_t row = 0; row < rows; ++row) {
        max_abs = std::max(max_abs, std::fabs(host[row] - control[row]));
      }
      std::printf("layout_vs_control name=%s layout=%s max_abs=%.9g\n", name,
                  kLayouts[layout].id, static_cast<double>(max_abs));
      if (max_abs > std::max(kLayoutAbs, staged_gate)) rc = 1;
    }
    rc |= check_guards(&box, name);
  }
  std::printf("case=%s rows=%zu cols=%zu layouts=%d\n", name, rows, cols,
              kLayoutCount);
  free_case(&box);
  return rc;
}

int run_stage_reuse() {
  constexpr std::size_t rows = 17;
  constexpr std::size_t cols = 32;
  RunCase a;
  RunCase b;
  if (alloc_case(rows, cols, &a) != 0) return 1;
  if (alloc_case(rows, cols, &b) != 0) {
    free_case(&a);
    return 1;
  }
  fill_weights(rows, cols, "random", &a.host_w);
  fill_activation("random", cols, 0x11U, &a.host_a);
  fill_weights(rows, cols, "random", &b.host_w);
  fill_activation("cancel", cols, 0x22U, &b.host_a);
  if (upload_case(&a) != 0 || upload_case(&b) != 0) {
    free_case(&a);
    free_case(&b);
    return 1;
  }
  qw38::cuda::SchedulerWorkspace workspace;
  const qw38::Status status = workspace.create(256);
  if (!status.is_ok()) {
    free_case(&a);
    free_case(&b);
    return fail_status(status);
  }
  const Layout layout = kLayouts[0];
  int stages = 0;
  const __nv_bfloat16* cookie = nullptr;
  std::size_t cookie_cols = 0;
  auto proj = [&](RunCase* box) -> cudaError_t {
    cudaError_t error = cudaSuccess;
    if (cookie != box->d_a || cookie_cols != box->cols) {
      error = qw38::cuda::launch_quantize_bf16_q8_1(box->d_a, workspace.q8_,
                                                    box->cols, nullptr);
      cookie = box->d_a;
      cookie_cols = box->cols;
      ++stages;
    }
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_q8_coop_mmv_prequant(
          box->d_w + kGuard, box->rows, box->cols, workspace.q8_, box->d_o,
          layout.rows_per_cta, layout.warps_per_row, nullptr);
    }
    return error;
  };
  cudaError_t error = proj(&a);
  if (error == cudaSuccess) error = proj(&a);
  if (error == cudaSuccess) error = proj(&a);
  const int reused = stages;
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  std::vector<float> stale(rows, 0.0F);
  if (error == cudaSuccess) {
    error = cudaMemcpy(stale.data(), a.d_o, rows * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) {
    free_case(&a);
    free_case(&b);
    return fail_cuda("reuse", error);
  }
  error = cudaMemcpy(a.d_a, b.host_a.data(), cols * sizeof(__nv_bfloat16),
                     cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    free_case(&a);
    free_case(&b);
    return fail_cuda("overwrite", error);
  }
  workspace.q8_decode_staged_activation_ = a.d_a;
  workspace.q8_decode_staged_columns_ = cols;
  error = proj(&a);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  std::vector<float> without(rows, 0.0F);
  if (error == cudaSuccess) {
    error = cudaMemcpy(without.data(), a.d_o, rows * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  workspace.invalidate_q8_decode_staging();
  if (workspace.q8_decode_staged_activation_ != nullptr ||
      workspace.q8_decode_staged_columns_ != 0) {
    std::fprintf(stderr, "invalidate_q8_decode_staging left cookie set\n");
    free_case(&a);
    free_case(&b);
    return 1;
  }
  cookie = nullptr;
  cookie_cols = 0;
  error = proj(&a);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  std::vector<float> with(rows, 0.0F);
  if (error == cudaSuccess) {
    error = cudaMemcpy(with.data(), a.d_o, rows * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  const int after = stages;
  std::printf(
      "stage_reuse first=%d after_overwrite_no_invalidate=%d "
      "after_invalidate=%d cookie_null_after=%s\n",
      reused, reused, after,
      json_bool(workspace.q8_decode_staged_activation_ == nullptr));
  int rc = 0;
  if (reused != 1 || after != 2) {
    std::fprintf(stderr, "expected exact stage counts 1 then 2, got %d %d\n",
                 reused, after);
    rc = 1;
  }
  float stale_vs_without = 0.0F;
  float with_vs_without = 0.0F;
  for (std::size_t row = 0; row < rows; ++row) {
    stale_vs_without =
        std::max(stale_vs_without, std::fabs(stale[row] - without[row]));
    with_vs_without =
        std::max(with_vs_without, std::fabs(with[row] - without[row]));
  }
  std::printf("overwrite pointer_identity_not_data stale_vs_without=%.9g "
              "invalidate_changed=%.9g\n",
              static_cast<double>(stale_vs_without),
              static_cast<double>(with_vs_without));
  if (stale_vs_without > kLayoutAbs) rc = 1;
  if (with_vs_without <= kLayoutAbs) {
    std::fprintf(stderr, "invalidate did not change overwritten result\n");
    rc = 1;
  }
  free_case(&a);
  free_case(&b);
  return error == cudaSuccess ? rc : fail_cuda("stage reuse", error);
}

int run_smoke() {
  int rc = run_layouts_vs_ref("m17_k32", 17, 32, "random", "random", 0xA11CE5u,
                              true, kStagedAbs, true);
  for (int i = 0; i < kLayoutCount; ++i) {
    int registers = 0;
    std::size_t local_bytes = 0;
    int occupancy = 0;
    qw38::cuda::q8_coop_kernel_attributes(kLayouts[i].rows_per_cta,
                                          kLayouts[i].warps_per_row, &registers,
                                          &local_bytes, &occupancy);
    std::printf("attrs layout=%s rows=%u warps=%u regs=%d local=%zu occ=%d\n",
                kLayouts[i].id, kLayouts[i].rows_per_cta,
                kLayouts[i].warps_per_row, registers, local_bytes, occupancy);
    if (local_bytes != 0) rc = 1;
  }
  return rc;
}

int load_model(const char* path, qw38::internal::MappedFile* mapping,
               qw38::internal::ModelInfo* info,
               qw38::internal::ModelWeights* weights,
               qw38::cuda::ResidentModel* model) {
  qw38::Status status = qw38::internal::inspect_gguf(path, info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(info);
  if (status.is_ok()) status = mapping->open(path);
  if (status.is_ok()) {
    status = qw38::internal::bind_model_weights(*info, *mapping, weights);
  }
  if (status.is_ok()) {
    status = model->upload(*weights, mapping->data(), mapping->size());
  }
  if (!status.is_ok()) return fail_status(status);
  return 0;
}

int copy_tensor_rows(const qw38::cuda::DeviceTensor& tensor,
                     const std::vector<std::size_t>& rows,
                     std::vector<std::uint8_t>* host) {
  const std::size_t row_bytes = (tensor.columns / kQ80Values) * kQ80Bytes;
  host->assign(rows.size() * row_bytes, 0);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const cudaError_t error = cudaMemcpy(
        host->data() + i * row_bytes, tensor.data + rows[i] * row_bytes,
        row_bytes, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) return fail_cuda("copy tensor rows", error);
  }
  return 0;
}

int run_sampled_real(const qw38::cuda::ResidentModel&, std::size_t layer,
                     const qw38::cuda::DeviceTensor& tensor, const char* name) {
  const std::vector<std::size_t> rows = sample_rows(tensor.rows);
  std::vector<std::uint8_t> packed;
  if (copy_tensor_rows(tensor, rows, &packed) != 0) return 1;
  int rc = 0;
  constexpr const char* kPatterns[kCaptures] = {"random", "zero", "cancel",
                                                "extremes"};
  for (int capture = 0; capture < kCaptures; ++capture) {
    RunCase box;
    if (alloc_case(rows.size(), tensor.columns, &box) != 0) return 1;
    box.host_w = packed;
    fill_activation(kPatterns[capture], tensor.columns,
                    0x50U + static_cast<std::uint32_t>(capture) * 17U,
                    &box.host_a);
    if (upload_case(&box) != 0) {
      free_case(&box);
      return 1;
    }
    cudaError_t error = qw38::cuda::launch_quantize_bf16_q8_1(
        box.d_a, box.d_s, tensor.columns, nullptr);
    if (error != cudaSuccess) {
      free_case(&box);
      return fail_cuda("sampled quant", error);
    }
    std::vector<std::uint8_t> staged_bytes(q81_bytes(tensor.columns), 0);
    error = cudaMemcpy(staged_bytes.data(), box.d_s, staged_bytes.size(),
                       cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
      free_case(&box);
      return fail_cuda("sampled staged", error);
    }
    std::vector<double> staged;
    host_from_gpu_q8_1(staged_bytes.data(), tensor.columns, &staged);
    const std::vector<std::size_t> local = all_rows(rows.size());
    std::vector<double> ref;
    if (!host_fp64(box.host_w, rows.size(), tensor.columns, staged, local,
                   &ref)) {
      free_case(&box);
      return 1;
    }
    for (int layout = 0; layout < kLayoutCount; ++layout) {
      error = launch_layout(kLayouts[layout], box.d_w + kGuard, rows.size(),
                            tensor.columns, box.d_s, box.d_o);
      if (error == cudaSuccess) error = cudaDeviceSynchronize();
      if (error != cudaSuccess) {
        free_case(&box);
        return fail_cuda("sampled launch", error);
      }
      std::vector<float> host(rows.size(), 0.0F);
      error = cudaMemcpy(host.data(), box.d_o, host.size() * sizeof(float),
                         cudaMemcpyDeviceToHost);
      if (error != cudaSuccess) {
        free_case(&box);
        return fail_cuda("sampled d2h", error);
      }
      char label[128];
      std::snprintf(label, sizeof(label), "%s_layer%zu_cap%d_%s", name, layer,
                    capture, kLayouts[layout].id);
      rc |= compare_rows(host, ref, local, abs_gate(tensor.columns), label);
      rc |= check_guards(&box, label);
    }
    free_case(&box);
  }
  return rc;
}

int run_correctness(const char* model_path) {
  int rc = run_layouts_vs_ref("m1_k32", 1, 32, "random", "random", 0x1U, true,
                              kStagedAbs, true);
  rc |= run_layouts_vs_ref("m7_k64", 7, 64, "random", "random", 0x7U, true,
                           kStagedAbs, true);
  rc |= run_layouts_vs_ref("m17_k512", 17, 512, "random", "random", 0x11U, true,
                           kStagedAbs, true);
  rc |= run_layouts_vs_ref("m48_k5120", 48, 5120, "random", "random", 0x30U,
                           false, kStagedAbs, false);
  rc |= run_layouts_vs_ref("m17_k32_extremes", 17, 32, "extremes", "extremes",
                           0xE1U, true, kStagedAbs, true);
  rc |= run_layouts_vs_ref("m17_k32_zero", 17, 32, "random", "zero", 0x0U, true,
                           kStagedAbs, true);
  rc |= run_layouts_vs_ref("m17_k32_cancel", 17, 32, "random", "cancel", 0xC1U,
                           true, kStagedAbs, true);
  rc |= run_stage_reuse();
  if (model_path == nullptr || model_path[0] == '\0') {
    std::fprintf(stderr, "correctness requires the pinned GGUF\n");
    return 1;
  }
  qw38::internal::MappedFile mapping;
  qw38::internal::ModelInfo info;
  qw38::internal::ModelWeights weights;
  qw38::cuda::ResidentModel model;
  if (load_model(model_path, &mapping, &info, &weights, &model) != 0) return 1;
  const std::size_t held_out[2] = {62, 63};
  for (std::size_t layer : held_out) {
    const qw38::cuda::DeviceLayer& slot = model.layer(layer);
    if (slot.kind == qw38::internal::LayerKind::kGdn) {
      rc |= run_sampled_real(model, layer, slot.gdn.packed_qkv, "gdn_qkv");
      rc |= run_sampled_real(model, layer, slot.gdn.output, "gdn_out");
    } else {
      rc |= run_sampled_real(model, layer, slot.attention.query_gate,
                             "attn_qg");
    }
  }
  return rc;
}

cudaError_t time_group(const Layout& layout, bool gdn_input, bool gdn_output,
                       bool attn, const qw38::cuda::DeviceLayer& layer,
                       __nv_bfloat16* act5120, __nv_bfloat16* act6144,
                       void* staged, float* o0, float* o1, float* o2, float* o3,
                       cudaEvent_t start, cudaEvent_t stop, float* ms) {
  cudaError_t error = cudaEventRecord(start);
  if (gdn_input && error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_bf16_q8_1(act5120, staged, 5120,
                                                  nullptr);
    if (error == cudaSuccess) {
      error = launch_layout(layout, layer.gdn.packed_qkv.data,
                            layer.gdn.packed_qkv.rows,
                            layer.gdn.packed_qkv.columns, staged, o0);
    }
    if (error == cudaSuccess) {
      error = launch_layout(layout, layer.gdn.value_gate.data,
                            layer.gdn.value_gate.rows,
                            layer.gdn.value_gate.columns, staged, o1);
    }
    if (error == cudaSuccess) {
      error = launch_layout(layout, layer.gdn.alpha.data, layer.gdn.alpha.rows,
                            layer.gdn.alpha.columns, staged, o2);
    }
    if (error == cudaSuccess) {
      error = launch_layout(layout, layer.gdn.beta.data, layer.gdn.beta.rows,
                            layer.gdn.beta.columns, staged, o3);
    }
  }
  if (gdn_output && error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_bf16_q8_1(act6144, staged, 6144,
                                                  nullptr);
    if (error == cudaSuccess) {
      error = launch_layout(layout, layer.gdn.output.data, layer.gdn.output.rows,
                            layer.gdn.output.columns, staged, o0);
    }
  }
  if (attn && error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_bf16_q8_1(act5120, staged, 5120,
                                                  nullptr);
    if (error == cudaSuccess) {
      error = launch_layout(layout, layer.attention.query_gate.data,
                            layer.attention.query_gate.rows,
                            layer.attention.query_gate.columns, staged, o0);
    }
    if (error == cudaSuccess) {
      error = launch_layout(layout, layer.attention.key.data,
                            layer.attention.key.rows,
                            layer.attention.key.columns, staged, o1);
    }
    if (error == cudaSuccess) {
      error = launch_layout(layout, layer.attention.value.data,
                            layer.attention.value.rows,
                            layer.attention.value.columns, staged, o2);
    }
  }
  if (error == cudaSuccess) error = cudaEventRecord(stop);
  if (error == cudaSuccess) error = cudaEventSynchronize(stop);
  if (error == cudaSuccess) error = cudaEventElapsedTime(ms, start, stop);
  return error;
}

struct GroupMean {
  float rotating = 0.0F;
  float hot = 0.0F;
};

int collect_gdn_attn(const qw38::cuda::ResidentModel& model,
                     std::vector<std::size_t>* gdn,
                     std::vector<std::size_t>* attn) {
  for (std::size_t layer = 0; layer < model.layer_count(); ++layer) {
    if (model.layer(layer).kind == qw38::internal::LayerKind::kGdn) {
      gdn->push_back(layer);
    } else {
      attn->push_back(layer);
    }
  }
  return gdn->size() == kGdnLayers && attn->size() == kAttnLayers ? 0 : 1;
}

int run_screen(const char* model_path, bool acceptance) {
  if (model_path == nullptr || model_path[0] == '\0') {
    std::fprintf(stderr, "screen/acceptance require the pinned GGUF\n");
    return 1;
  }
  qw38::internal::MappedFile mapping;
  qw38::internal::ModelInfo info;
  qw38::internal::ModelWeights weights;
  qw38::cuda::ResidentModel model;
  if (load_model(model_path, &mapping, &info, &weights, &model) != 0) return 1;
  std::vector<std::size_t> gdn;
  std::vector<std::size_t> attn;
  if (collect_gdn_attn(model, &gdn, &attn) != 0) {
    std::fprintf(stderr, "unexpected GDN/attention layer counts\n");
    return 1;
  }
  std::vector<__nv_bfloat16> h5120;
  std::vector<__nv_bfloat16> h6144;
  fill_activation("random", 5120, 0x51U, &h5120);
  fill_activation("random", 6144, 0x61U, &h6144);
  __nv_bfloat16* act5120 = nullptr;
  __nv_bfloat16* act6144 = nullptr;
  void* staged = nullptr;
  float* o0 = nullptr;
  float* o1 = nullptr;
  float* o2 = nullptr;
  float* o3 = nullptr;
  cudaError_t error = cudaMalloc(&act5120, 5120 * sizeof(__nv_bfloat16));
  if (error == cudaSuccess) {
    error = cudaMalloc(&act6144, 6144 * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&staged, q81_bytes(6144));
  }
  if (error == cudaSuccess) error = cudaMalloc(&o0, 12288 * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&o1, 6144 * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&o2, 1024 * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&o3, 48 * sizeof(float));
  if (error == cudaSuccess) {
    error = cudaMemcpy(act5120, h5120.data(), 5120 * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(act6144, h6144.data(), 6144 * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) return fail_cuda("screen alloc", error);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error != cudaSuccess) return fail_cuda("events", error);

  const int warmups = 1;
  const int samples = 3;
  GroupMean means[kLayoutCount][3]{};
  auto time_mode = [&](int layout, int group, bool rotating, bool warmup,
                       int sample, float* acc, int* n) -> cudaError_t {
    const bool gdn_input = group == 0;
    const bool gdn_output = group == 1;
    const bool attn_group = group == 2;
    std::size_t layer = 0;
    if (gdn_input || gdn_output) {
      layer = rotating ? gdn[static_cast<std::size_t>(sample) % gdn.size()]
                       : gdn[0];
    } else {
      layer = rotating ? attn[static_cast<std::size_t>(sample) % attn.size()]
                       : attn[0];
    }
    float ms = 0.0F;
    const cudaError_t local = time_group(
        kLayouts[layout], gdn_input, gdn_output, attn_group, model.layer(layer),
        act5120, act6144, staged, o0, o1, o2, o3, start, stop, &ms);
    if (local != cudaSuccess) return local;
    std::printf("screen group=%d layout=%s rotating=%s warmup=%s sample=%d "
                "layer=%zu ms=%.9g\n",
                group, kLayouts[layout].id, json_bool(rotating),
                json_bool(warmup), sample, layer, static_cast<double>(ms));
    if (!warmup) {
      *acc += ms;
      ++*n;
    }
    return cudaSuccess;
  };

  for (int group = 0; group < 3 && error == cudaSuccess; ++group) {
    for (int rotating = 1; rotating >= 0 && error == cudaSuccess; --rotating) {
      for (int w = 0; w < warmups && error == cudaSuccess; ++w) {
        for (int layout = 0; layout < kLayoutCount && error == cudaSuccess;
             ++layout) {
          float ignore = 0.0F;
          int n = 0;
          const int order = (w + rotating) % 2 == 0 ? layout
                                                    : kLayoutCount - 1 - layout;
          error = time_mode(order, group, rotating != 0, true, w, &ignore, &n);
        }
      }
      float acc[kLayoutCount]{};
      int n[kLayoutCount]{};
      for (int sample = 0; sample < samples && error == cudaSuccess; ++sample) {
        for (int step = 0; step < kLayoutCount && error == cudaSuccess;
             ++step) {
          const int layout =
              sample % 2 == 0 ? step : kLayoutCount - 1 - step;
          error = time_mode(layout, group, rotating != 0, false, sample,
                            &acc[layout], &n[layout]);
        }
      }
      for (int layout = 0; layout < kLayoutCount; ++layout) {
        const float mean =
            n[layout] == 0 ? 0.0F : acc[layout] / static_cast<float>(n[layout]);
        if (rotating != 0) means[layout][group].rotating = mean;
        else means[layout][group].hot = mean;
      }
    }
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(act5120);
  cudaFree(act6144);
  cudaFree(staged);
  cudaFree(o0);
  cudaFree(o1);
  cudaFree(o2);
  cudaFree(o3);
  if (error != cudaSuccess) return fail_cuda("screen time", error);

  int winner = 0;
  double best = 1.0e30;
  double control = 0.0;
  for (int layout = 0; layout < kLayoutCount; ++layout) {
    const double weighted =
        static_cast<double>(kGdnLayers) *
            (means[layout][0].rotating + means[layout][1].rotating) +
        static_cast<double>(kAttnLayers) * means[layout][2].rotating;
    int registers = 0;
    std::size_t local_bytes = 0;
    int occupancy = 0;
    qw38::cuda::q8_coop_kernel_attributes(kLayouts[layout].rows_per_cta,
                                          kLayouts[layout].warps_per_row,
                                          &registers, &local_bytes, &occupancy);
    std::printf(
        "weighted layout=%s rotating_ms=%.9g hot_ms=%.9g gdn_in_r=%.9g "
        "gdn_out_r=%.9g attn_r=%.9g gdn_in_h=%.9g gdn_out_h=%.9g attn_h=%.9g "
        "regs=%d local=%zu occ=%d\n",
        kLayouts[layout].id, weighted,
        static_cast<double>(kGdnLayers) *
                (means[layout][0].hot + means[layout][1].hot) +
            static_cast<double>(kAttnLayers) * means[layout][2].hot,
        static_cast<double>(means[layout][0].rotating),
        static_cast<double>(means[layout][1].rotating),
        static_cast<double>(means[layout][2].rotating),
        static_cast<double>(means[layout][0].hot),
        static_cast<double>(means[layout][1].hot),
        static_cast<double>(means[layout][2].hot), registers, local_bytes,
        occupancy);
    if (layout == 0) control = weighted;
    if (weighted < best) {
      best = weighted;
      winner = layout;
    }
  }
  const double saving = control - best;
  const bool keep = winner != 0 && saving > 0.0;
  const bool effort = saving >= 0.10;
  std::printf(
      "screen_winner=%s control=%s control_ms=%.9g winner_ms=%.9g "
      "saving_ms=%.9g keep=%s effort_0p10=%s installed=%s "
      "layout_per_role skinny=%s medium=%s wide=%s\n",
      kLayouts[winner].id, kLayouts[0].id, control, best, saving,
      json_bool(keep), json_bool(effort), json_bool(false),
      keep ? kLayouts[winner].id : kLayouts[0].id,
      keep ? kLayouts[winner].id : kLayouts[0].id,
      keep ? kLayouts[winner].id : kLayouts[0].id);
  std::printf(
      "selected_q8_decode_path=%s historical_warps=4/4/4 "
      "rows=%u/%u/%u layout_warps=%u/%u/%u\n",
      qw38::cuda::selected_q8_decode_path(),
      qw38::cuda::selected_q8_decode_rows_skinny(),
      qw38::cuda::selected_q8_decode_rows_medium(),
      qw38::cuda::selected_q8_decode_rows_wide(),
      qw38::cuda::selected_q8_decode_layout_warps_skinny(),
      qw38::cuda::selected_q8_decode_layout_warps_medium(),
      qw38::cuda::selected_q8_decode_layout_warps_wide());
  (void)acceptance;
  return 0;
}

void emit_payload(const char* workload, const char* status) {
  const qw38::cuda::Q8DecodeLayout skinny =
      qw38::cuda::q8_decode_layout_for_rows(48);
  const qw38::cuda::Q8DecodeLayout wide =
      qw38::cuda::q8_decode_layout_for_rows(10240);
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-064\",\"workload\":\"%s\","
      "\"tier\":\"%s\",\"status\":\"%s\",\"selected_q8_decode_path\":\"%s\","
      "\"selected_rows_skinny\":%u,\"selected_warps_skinny\":%u,"
      "\"selected_rows_wide\":%u,\"selected_warps_wide\":%u,"
      "\"historical_warps_skinny\":%u,\"claims_throughput\":false,"
      "\"llama_revision\":\"%s\",\"gguf_sha256\":\"%s\"}\n",
      kPrefix, workload, qw38::cuda::test_tier_name(), status,
      qw38::cuda::selected_q8_decode_path(), skinny.rows_per_cta,
      skinny.warps_per_row, wide.rows_per_cta, wide.warps_per_row,
      qw38::cuda::selected_q8_decode_warps_skinny(), kLlamaRev, kGgufSha);
}

}  // namespace

int main(int argc, char** argv) {
  if (!qw38::cuda::test_tier_valid()) {
    std::fprintf(stderr, "QW38_CUDA_TEST_TIER must be set\n");
    return 2;
  }
  Options options;
  if (parse_args(argc, argv, &options) != 0) return 2;
  const char* workload = options.workload != nullptr
                             ? options.workload
                             : default_workload(qw38::cuda::test_tier());
  int rc = 0;
  if (std::strcmp(workload, "smoke") == 0) {
    rc = run_smoke();
  } else if (std::strcmp(workload, "correctness") == 0) {
    rc = run_correctness(options.model);
  } else if (std::strcmp(workload, "screen") == 0 ||
             std::strcmp(workload, "acceptance") == 0) {
    rc = run_screen(options.model, std::strcmp(workload, "acceptance") == 0);
  } else {
    return usage(argv[0]);
  }
  emit_payload(workload, rc == 0 ? "passed" : "failed");
  std::printf("status=%s\n", rc == 0 ? "passed" : "failed");
  return rc;
}
