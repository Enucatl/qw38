#include "full_scheduler.h"
#include "mixer.h"
#include "model.h"
#include "quant.h"
#include "quant_mmv.h"
#include "test_tier.h"
#include "weights.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr char kPrefix[] = "QW38_OPT065_MMQ_TILES_RESULT=";
constexpr const char* kGgufSha =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr const char* kLlamaRev =
    "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr std::size_t kQ4KBytes = 144;
constexpr std::size_t kQ4KValues = 256;
constexpr std::size_t kQ8Bytes = 34;
constexpr std::size_t kQ8Values = 32;
constexpr float kAbsScale = 0.20F;
constexpr float kRelTol = 0.05F;
constexpr float kOpt044Abs = 3.0e-4F;
constexpr int kSampledOut = 16;
constexpr int kSampledPrompt = 8;
constexpr int kProdPrompt = 4;

struct Tile {
  const char* id;
  unsigned int quality_i;
  unsigned int prompt_tile;
};

constexpr Tile kTiles[] = {
    {"i128_j128", 128, 128},
    {"i64_j128", 64, 128},
    {"i128_j64", 128, 64},
    {"i64_j64", 64, 64},
};
constexpr int kTileCount = 4;
constexpr int kControl = 0;

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

void fill_q4_weights(std::size_t rows, std::size_t columns,
                     std::vector<std::uint8_t>* weights) {
  weights->assign(rows * (columns / kQ4KValues) * kQ4KBytes, 0);
  for (std::size_t index = 0; index < weights->size(); ++index) {
    (*weights)[index] = static_cast<std::uint8_t>((index * 73U + 19U) & 0xFFU);
  }
  for (std::size_t offset = 0; offset < weights->size(); offset += kQ4KBytes) {
    write_u16(weights->data() + offset, 0x2400U);
    write_u16(weights->data() + offset + 2, 0x1C00U);
  }
}

void fill_unique_q4_weights(std::size_t rows, std::size_t columns,
                            std::vector<std::uint8_t>* weights) {
  fill_q4_weights(rows, columns, weights);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t block = 0; block < columns / kQ4KValues; ++block) {
      std::uint8_t* packed =
          weights->data() + (row * (columns / kQ4KValues) + block) * kQ4KBytes;
      packed[16 + (row % 32)] =
          static_cast<std::uint8_t>((row + block * 17U) & 0x7FU);
    }
  }
}

void fill_q8_weights(std::size_t rows, std::size_t columns,
                     std::vector<std::uint8_t>* weights) {
  weights->assign(rows * (columns / kQ8Values) * kQ8Bytes, 0);
  for (std::size_t index = 0; index < weights->size(); ++index) {
    (*weights)[index] = static_cast<std::uint8_t>((index * 41U + 7U) & 0xFFU);
  }
  for (std::size_t offset = 0; offset < weights->size(); offset += kQ8Bytes) {
    write_u16(weights->data() + offset, 0x3C00U);
  }
}

void fill_prompt(std::size_t prompt_rows, std::size_t columns,
                 std::vector<__nv_bfloat16>* prompt, std::uint32_t seed) {
  prompt->resize(prompt_rows * columns);
  for (std::size_t index = 0; index < prompt->size(); ++index) {
    (*prompt)[index] = __float2bfloat16_rn(
        unit_normal(static_cast<std::uint32_t>(index), seed));
  }
}

bool reference_dequant_gemm(const std::vector<std::uint8_t>& weights,
                            std::size_t output_rows, std::size_t columns,
                            const std::vector<__nv_bfloat16>& prompt,
                            std::size_t prompt_rows,
                            const std::vector<std::size_t>& out_sample,
                            const std::vector<std::size_t>& prompt_sample,
                            std::vector<float>* output) {
  output->assign(prompt_rows * output_rows,
                 std::numeric_limits<float>::quiet_NaN());
  std::vector<float> decoded(kQ4KValues);
  std::vector<float> decoded_row(columns);
  for (std::size_t out : out_sample) {
    for (std::size_t block = 0; block < columns / kQ4KValues; ++block) {
      const std::uint8_t* packed =
          weights.data() +
          (out * (columns / kQ4KValues) + block) * kQ4KBytes;
      const qw38::Status status = qw38::internal::decode_q4_k(
          packed, kQ4KBytes, decoded.data(), decoded.size());
      if (!status.is_ok()) return false;
      std::copy(decoded.begin(), decoded.end(),
                decoded_row.begin() + block * kQ4KValues);
    }
    for (std::size_t prompt_row : prompt_sample) {
      float sum = 0.0F;
      for (std::size_t column = 0; column < columns; ++column) {
        sum += decoded_row[column] *
               __bfloat162float(prompt[prompt_row * columns + column]);
      }
      (*output)[prompt_row * output_rows + out] = sum;
    }
  }
  return true;
}

std::vector<std::size_t> all_indices(std::size_t count) {
  std::vector<std::size_t> indices(count);
  for (std::size_t index = 0; index < count; ++index) indices[index] = index;
  return indices;
}

std::vector<std::size_t> sample_indices(std::size_t count, int take) {
  std::vector<std::size_t> indices;
  if (count == 0) return indices;
  const int n = std::min(take, static_cast<int>(count));
  indices.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    indices.push_back(static_cast<std::size_t>(i) * (count - 1) /
                      static_cast<std::size_t>(std::max(n - 1, 1)));
  }
  if (indices.back() != count - 1) indices.back() = count - 1;
  return indices;
}

bool ds4_ok(const std::vector<float>& got, const std::vector<float>& ref,
            std::size_t columns, float* max_abs, std::size_t* bad,
            std::size_t* nonfinite) {
  const float abs_tol = kAbsScale * std::sqrt(static_cast<float>(columns));
  *max_abs = 0.0F;
  *bad = 0;
  *nonfinite = 0;
  for (std::size_t index = 0; index < got.size(); ++index) {
    if (std::isnan(ref[index])) continue;
    if (!std::isfinite(got[index]) || !std::isfinite(ref[index])) {
      ++*nonfinite;
      continue;
    }
    const float absolute = std::fabs(got[index] - ref[index]);
    const float relative = ref[index] != 0.0F
                               ? absolute / std::fabs(ref[index])
                               : (absolute > 0.0F ? INFINITY : 0.0F);
    *max_abs = std::max(*max_abs, absolute);
    if (absolute > abs_tol && relative > kRelTol) ++*bad;
  }
  return *nonfinite == 0 && *bad == 0;
}

int launch_q4(const std::uint8_t* weights, std::size_t output_rows,
              std::size_t columns, const qw38::cuda::Q8Block* y,
              std::size_t prompt_rows, float* output, const Tile& tile) {
  qw38::cuda::clear_mmq_tile_dispatch();
  const cudaError_t error = qw38::cuda::launch_quant_mmq_mma_y_ij(
      qw38::cuda::QuantKind::kQ4K, weights, output_rows, columns, y, prompt_rows,
      output, tile.quality_i, tile.prompt_tile, nullptr);
  if (error != cudaSuccess) return fail_cuda("q4 launch", error);
  return 0;
}

int check_dispatch(const Tile& tile, bool want_pipeline, const char* label) {
  const qw38::cuda::MmqTileDispatch rec = qw38::cuda::last_mmq_tile_dispatch();
  const bool ok_ident = std::strcmp(rec.ident, tile.id) == 0;
  const bool ok_ij = rec.quality_i == tile.quality_i &&
                     rec.prompt_tile == tile.prompt_tile;
  const bool ok_pipe = rec.pipeline == want_pipeline;
  const bool ok_fma = !want_pipeline || (rec.fma && rec.async_y);
  const bool ok_path = std::strcmp(rec.path, "fma_async") == 0;
  std::printf(
      "dispatch %s tile=%s ident=%s kernel=%s aligned=%s pipeline=%s "
      "fallback=%s fma=%s async_y=%s path=%s\n",
      label, tile.id, rec.ident, rec.kernel, json_bool(rec.aligned),
      json_bool(rec.pipeline), json_bool(rec.fallback), json_bool(rec.fma),
      json_bool(rec.async_y), rec.path);
  if (!ok_ident || !ok_ij || !ok_pipe || !ok_fma || !ok_path) {
    std::fprintf(stderr, "dispatch mismatch for %s %s\n", label, tile.id);
    return 1;
  }
  return 0;
}

int run_q4_case(const char* label, std::size_t output_rows, std::size_t columns,
                std::size_t prompt_rows, const Tile* tiles, int tile_count,
                bool unique, bool full_ref) {
  std::vector<std::uint8_t> weights;
  if (unique) {
    fill_unique_q4_weights(output_rows, columns, &weights);
  } else {
    fill_q4_weights(output_rows, columns, &weights);
  }
  std::vector<__nv_bfloat16> prompt;
  fill_prompt(prompt_rows, columns, &prompt, 0xA11CE5u);
  const std::vector<std::size_t> out_s =
      full_ref ? all_indices(output_rows)
               : sample_indices(output_rows, kSampledOut);
  const std::vector<std::size_t> pr_s =
      full_ref ? all_indices(prompt_rows)
               : sample_indices(prompt_rows, kSampledPrompt);
  std::vector<float> expected;
  if (!reference_dequant_gemm(weights, output_rows, columns, prompt, prompt_rows,
                              out_s, pr_s, &expected)) {
    std::fprintf(stderr, "host GEMM failed for %s\n", label);
    return 1;
  }

  std::uint8_t* d_w = nullptr;
  __nv_bfloat16* d_p = nullptr;
  qw38::cuda::Q8Block* d_y = nullptr;
  float* d_o = nullptr;
  cudaError_t error = cudaMalloc(&d_w, weights.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&d_p, prompt.size() * sizeof(prompt[0]));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&d_y, qw38::cuda::q8_prompt_workspace_bytes(prompt_rows,
                                                                  columns));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&d_o, prompt_rows * output_rows * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(d_w, weights.data(), weights.size(),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(d_p, prompt.data(), prompt.size() * sizeof(prompt[0]),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_mmq_q8_1(
        qw38::cuda::QuantKind::kQ4K, d_p, prompt_rows, columns, d_y, nullptr);
  }
  if (error != cudaSuccess) {
    cudaFree(d_o);
    cudaFree(d_y);
    cudaFree(d_p);
    cudaFree(d_w);
    return fail_cuda("q4 case alloc", error);
  }

  int rc = 0;
  std::vector<float> actual(prompt_rows * output_rows);
  for (int t = 0; t < tile_count && rc == 0; ++t) {
    const Tile& tile = tiles[t];
    const bool aligned = output_rows % tile.quality_i == 0 &&
                         prompt_rows % tile.prompt_tile == 0;
    rc = launch_q4(d_w, output_rows, columns, d_y, prompt_rows, d_o, tile);
    if (rc == 0) {
      error = cudaDeviceSynchronize();
      if (error != cudaSuccess) rc = fail_cuda("q4 sync", error);
    }
    if (rc == 0) {
      error = cudaMemcpy(actual.data(), d_o, actual.size() * sizeof(float),
                         cudaMemcpyDeviceToHost);
      if (error != cudaSuccess) rc = fail_cuda("q4 d2h", error);
    }
    float max_abs = 0.0F;
    std::size_t bad = 0;
    std::size_t nonfinite = 0;
    const bool ok = rc == 0 && ds4_ok(actual, expected, columns, &max_abs, &bad,
                                      &nonfinite);
    std::printf(
        "case=%s tile=%s m=%zu k=%zu n=%zu aligned=%s max_abs=%.9g bad=%zu "
        "nonfinite=%zu\n",
        label, tile.id, output_rows, columns, prompt_rows, json_bool(aligned),
        max_abs, bad, nonfinite);
    if (rc == 0) rc = check_dispatch(tile, aligned, label);
    if (!ok) {
      std::fprintf(stderr, "quality failed for %s %s\n", label, tile.id);
      rc = 1;
    }
  }
  cudaFree(d_o);
  cudaFree(d_y);
  cudaFree(d_p);
  cudaFree(d_w);
  return rc;
}

int run_resource_table() {
  int rc = 0;
  std::printf("resource_table path=fma_async\n");
  for (int t = 0; t < kTileCount; ++t) {
    const Tile& tile = kTiles[t];
    int occupancy = 0;
    int registers = 0;
    std::size_t local_bytes = 0;
    std::size_t shared_bytes = 0;
    const cudaError_t error = qw38::cuda::mmq_pipeline_kernel_attributes(
        qw38::cuda::QuantKind::kQ4K, tile.prompt_tile, tile.quality_i,
        "fma_async", &occupancy, &registers, &local_bytes, &shared_bytes);
    const int occ2 = qw38::cuda::mmq_pipeline_occupancy(
        qw38::cuda::QuantKind::kQ4K, tile.prompt_tile, tile.quality_i,
        "fma_async");
    const std::size_t extra = qw38::cuda::mmq_pipeline_extra_shared_bytes(
        tile.prompt_tile, tile.quality_i, "fma_async");
    std::printf(
        "resource tile=%s occupancy=%d occupancy_helper=%d regs=%d local=%zu "
        "shared=%zu extra_y=%zu\n",
        tile.id, occupancy, occ2, registers, local_bytes, shared_bytes, extra);
    if (error != cudaSuccess || occupancy < 1 || occ2 != occupancy ||
        extra == 0) {
      std::fprintf(stderr, "resource query failed for %s\n", tile.id);
      rc = 1;
    }
  }
  return rc;
}

int run_q8_regression() {
  int rc = 0;
  const std::size_t cases[2][3] = {{128, 256, 128}, {32, 256, 128}};
  const char* names[2] = {"q8_large", "q8_skinny"};
  for (int c = 0; c < 2; ++c) {
    const std::size_t rows = cases[c][0];
    const std::size_t columns = cases[c][1];
    const std::size_t prompt_rows = cases[c][2];
    std::vector<std::uint8_t> weights;
    fill_q8_weights(rows, columns, &weights);
    std::vector<__nv_bfloat16> prompt;
    fill_prompt(prompt_rows, columns, &prompt, 0x51u);
    std::uint8_t* d_w = nullptr;
    __nv_bfloat16* d_p = nullptr;
    qw38::cuda::Q8Block* d_y = nullptr;
    float* d_o = nullptr;
    cudaError_t error = cudaMalloc(&d_w, weights.size());
    if (error == cudaSuccess) {
      error = cudaMalloc(&d_p, prompt.size() * sizeof(prompt[0]));
    }
    if (error == cudaSuccess) {
      error = cudaMalloc(&d_y, qw38::cuda::q8_prompt_workspace_bytes(
                                   prompt_rows, columns));
    }
    if (error == cudaSuccess) {
      error = cudaMalloc(&d_o, prompt_rows * rows * sizeof(float));
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(d_w, weights.data(), weights.size(),
                         cudaMemcpyHostToDevice);
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(d_p, prompt.data(), prompt.size() * sizeof(prompt[0]),
                         cudaMemcpyHostToDevice);
    }
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_quantize_mmq_q8_1(
          qw38::cuda::QuantKind::kQ8_0, d_p, prompt_rows, columns, d_y,
          nullptr);
    }
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_q8_mmq_quality_mma(d_w, rows, columns, d_y,
                                                    prompt_rows, d_o, nullptr);
    }
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    std::printf("q8_regression %s rows=%zu cols=%zu prompt=%zu launch=%s\n",
                names[c], rows, columns, prompt_rows,
                error == cudaSuccess ? "ok" : cudaGetErrorString(error));
    if (error != cudaSuccess) rc = 1;
    cudaFree(d_o);
    cudaFree(d_y);
    cudaFree(d_p);
    cudaFree(d_w);
  }
  if (qw38::cuda::launch_quant_mmq_mma_y_ij(
          qw38::cuda::QuantKind::kQ6K, nullptr, 256, 256, nullptr, 64, nullptr,
          64, 128, nullptr) != cudaErrorInvalidValue) {
    std::fprintf(stderr, "Q6 must reject I=64\n");
    rc = 1;
  }
  return rc;
}

int run_smoke() {
  int rc = run_resource_table();
  rc |= run_q4_case("smoke_m17", 17, 256, 17, kTiles, 2, true, true);
  return rc;
}

int copy_q4_rows(const qw38::cuda::DeviceTensor& tensor,
                 const std::vector<std::size_t>& rows,
                 std::vector<std::uint8_t>* host) {
  const std::size_t row_bytes = (tensor.columns / kQ4KValues) * kQ4KBytes;
  host->assign(rows.size() * row_bytes, 0);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const cudaError_t error = cudaMemcpy(
        host->data() + i * row_bytes, tensor.data + rows[i] * row_bytes,
        row_bytes, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) return fail_cuda("copy q4 rows", error);
  }
  return 0;
}

int run_production_sampled(const qw38::cuda::DeviceTensor& tensor,
                           const char* name) {
  const std::vector<std::size_t> rows = sample_indices(tensor.rows, kSampledOut);
  std::vector<std::uint8_t> packed;
  if (copy_q4_rows(tensor, rows, &packed) != 0) return 1;
  std::vector<__nv_bfloat16> prompt;
  fill_prompt(kProdPrompt, tensor.columns, &prompt, 0xC0FFEEu);
  std::vector<float> expected;
  if (!reference_dequant_gemm(packed, rows.size(), tensor.columns, prompt,
                              kProdPrompt, all_indices(rows.size()),
                              all_indices(kProdPrompt), &expected)) {
    return 1;
  }
  std::uint8_t* d_w = nullptr;
  __nv_bfloat16* d_p = nullptr;
  qw38::cuda::Q8Block* d_y = nullptr;
  float* d_o = nullptr;
  cudaError_t error = cudaMalloc(&d_w, packed.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&d_p, prompt.size() * sizeof(prompt[0]));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&d_y, qw38::cuda::q8_prompt_workspace_bytes(
                                 kProdPrompt, tensor.columns));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&d_o, kProdPrompt * rows.size() * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(d_w, packed.data(), packed.size(), cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(d_p, prompt.data(), prompt.size() * sizeof(prompt[0]),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_mmq_q8_1(
        qw38::cuda::QuantKind::kQ4K, d_p, kProdPrompt, tensor.columns, d_y,
        nullptr);
  }
  if (error != cudaSuccess) {
    cudaFree(d_o);
    cudaFree(d_y);
    cudaFree(d_p);
    cudaFree(d_w);
    return fail_cuda("prod alloc", error);
  }
  int rc = 0;
  std::vector<float> actual(kProdPrompt * rows.size());
  for (int t = 0; t < kTileCount && rc == 0; ++t) {
    rc = launch_q4(d_w, rows.size(), tensor.columns, d_y, kProdPrompt, d_o,
                   kTiles[t]);
    if (rc == 0) {
      error = cudaDeviceSynchronize();
      if (error != cudaSuccess) rc = fail_cuda("prod sync", error);
    }
    if (rc == 0) {
      error = cudaMemcpy(actual.data(), d_o, actual.size() * sizeof(float),
                         cudaMemcpyDeviceToHost);
      if (error != cudaSuccess) rc = fail_cuda("prod d2h", error);
    }
    float max_abs = 0.0F;
    std::size_t bad = 0;
    std::size_t nonfinite = 0;
    const bool ok =
        rc == 0 && ds4_ok(actual, expected, tensor.columns, &max_abs, &bad,
                          &nonfinite);
    std::printf(
        "production %s tile=%s rows=%zu k=%zu max_abs=%.9g bad=%zu "
        "nonfinite=%zu\n",
        name, kTiles[t].id, rows.size(), tensor.columns, max_abs, bad,
        nonfinite);
    if (!ok) rc = 1;
  }
  cudaFree(d_o);
  cudaFree(d_y);
  cudaFree(d_p);
  cudaFree(d_w);
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

int run_correctness(const char* model_path) {
  int rc = run_q4_case("corr_m65", 65, 256, 65, kTiles, kTileCount, false,
                       false);
  rc |= run_q4_case("corr_m129", 129, 512, 129, kTiles, kTileCount, false,
                    false);
  rc |= run_q4_case("tiny_full", 128, 256, 128, kTiles, kTileCount, true, true);
  const std::size_t tails[] = {1, 31, 32, 33};
  for (std::size_t n : tails) {
    char name[32];
    std::snprintf(name, sizeof(name), "tail_n%zu", n);
    rc |= run_q4_case(name, 65, 256, n, kTiles, 1, false, true);
  }
  rc |= run_q8_regression();
  if (model_path == nullptr || model_path[0] == '\0') {
    std::fprintf(stderr, "correctness requires the pinned GGUF\n");
    return 1;
  }
  qw38::internal::MappedFile mapping;
  qw38::internal::ModelInfo info;
  qw38::internal::ModelWeights weights;
  qw38::cuda::ResidentModel model;
  if (load_model(model_path, &mapping, &info, &weights, &model) != 0) return 1;
  const qw38::cuda::DeviceCommonLayer& layer = model.layer(0).common;
  rc |= run_production_sampled(layer.ffn_gate, "gate_k5120");
  rc |= run_production_sampled(layer.ffn_down, "down_k17408");
  return rc;
}

struct ShapeGroup {
  const char* id;
  std::size_t rows;
  std::size_t columns;
};

cudaError_t time_mmq(const std::uint8_t* weights, std::size_t rows,
                     std::size_t columns, const qw38::cuda::Q8Block* y,
                     std::size_t prompt_rows, float* output, const Tile& tile,
                     cudaEvent_t start, cudaEvent_t stop, float* ms) {
  cudaError_t error = cudaEventRecord(start, nullptr);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quant_mmq_mma_y_ij(
        qw38::cuda::QuantKind::kQ4K, weights, rows, columns, y, prompt_rows,
        output, tile.quality_i, tile.prompt_tile, nullptr);
  }
  if (error == cudaSuccess) error = cudaEventRecord(stop, nullptr);
  if (error == cudaSuccess) error = cudaEventSynchronize(stop);
  if (error == cudaSuccess) error = cudaEventElapsedTime(ms, start, stop);
  return error;
}

cudaError_t launch_complete_ffn(const qw38::cuda::DeviceCommonLayer& layer,
                                qw38::cuda::SchedulerWorkspace* workspace,
                                std::size_t prompt_rows, const Tile& gate,
                                const Tile& down) {
  qw38::cuda::FfnTileOverrideScope tiles(gate.quality_i, gate.prompt_tile,
                                         gate.quality_i, gate.prompt_tile,
                                         down.quality_i, down.prompt_tile);
  cudaError_t error = qw38::cuda::launch_quantize_mmq_q8_1(
      qw38::cuda::QuantKind::kQ4K, workspace->prompt_normalized_, prompt_rows,
      qw38::internal::kResidualWidth, workspace->prompt_q8_, nullptr);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quant_mmq_mma_y_ij(
        layer.ffn_gate.kind, layer.ffn_gate.data, layer.ffn_gate.rows,
        layer.ffn_gate.columns, workspace->prompt_q8_, prompt_rows,
        workspace->prompt_projection_a_,
        qw38::cuda::selected_ffn_gate_quality_i(),
        qw38::cuda::selected_ffn_gate_prompt_tile(), nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quant_mmq_mma_y_ij(
        layer.ffn_up.kind, layer.ffn_up.data, layer.ffn_up.rows,
        layer.ffn_up.columns, workspace->prompt_q8_, prompt_rows,
        workspace->prompt_projection_b_, qw38::cuda::selected_ffn_up_quality_i(),
        qw38::cuda::selected_ffn_up_prompt_tile(), nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_swiglu_quantize_mmq_q8_1(
        workspace->prompt_projection_a_, workspace->prompt_projection_b_,
        prompt_rows, qw38::internal::kFfnWidth, workspace->prompt_q8_, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quant_mmq_mma_y_ij(
        layer.ffn_down.kind, layer.ffn_down.data, layer.ffn_down.rows,
        layer.ffn_down.columns, workspace->prompt_q8_, prompt_rows,
        workspace->prompt_mixer_output_,
        qw38::cuda::selected_ffn_down_quality_i(),
        qw38::cuda::selected_ffn_down_prompt_tile(), nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_residual_add_fp32(
        workspace->prompt_residual_a_, workspace->prompt_mixer_output_,
        prompt_rows * qw38::internal::kResidualWidth,
        workspace->prompt_residual_b_, nullptr);
  }
  return error;
}

int run_screen(const char* model_path) {
  if (model_path == nullptr || model_path[0] == '\0') {
    std::fprintf(stderr, "screen requires the pinned GGUF\n");
    return 1;
  }
  qw38::internal::MappedFile mapping;
  qw38::internal::ModelInfo info;
  qw38::internal::ModelWeights weights;
  qw38::cuda::ResidentModel model;
  if (load_model(model_path, &mapping, &info, &weights, &model) != 0) return 1;
  const qw38::cuda::DeviceCommonLayer& layer = model.layer(0).common;
  constexpr std::size_t kPrompt = 4096;
  qw38::cuda::SchedulerWorkspace workspace;
  const qw38::Status status = workspace.create(kPrompt);
  if (!status.is_ok()) return fail_status(status);

  std::vector<__nv_bfloat16> prompt;
  fill_prompt(kPrompt, qw38::internal::kResidualWidth, &prompt, 0x4096u);
  std::vector<float> residual(kPrompt * qw38::internal::kResidualWidth);
  for (std::size_t index = 0; index < residual.size(); ++index) {
    residual[index] = unit_normal(static_cast<std::uint32_t>(index), 0xBEEFu);
  }
  cudaError_t error = cudaMemcpy(
      workspace.prompt_normalized_, prompt.data(),
      prompt.size() * sizeof(prompt[0]), cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(workspace.prompt_residual_a_, residual.data(),
                       residual.size() * sizeof(float), cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) return fail_cuda("screen upload", error);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error != cudaSuccess) return fail_cuda("events", error);

  const ShapeGroup groups[2] = {
      {"gate_up", layer.ffn_gate.rows, layer.ffn_gate.columns},
      {"down", layer.ffn_down.rows, layer.ffn_down.columns},
  };
  float means[2][kTileCount] = {};
  for (int g = 0; g < 2; ++g) {
    const ShapeGroup& group = groups[g];
    const std::uint8_t* w =
        g == 0 ? layer.ffn_gate.data : layer.ffn_down.data;
    const std::size_t y_cols =
        g == 0 ? qw38::internal::kResidualWidth : qw38::internal::kFfnWidth;
    const __nv_bfloat16* y_src = workspace.prompt_normalized_;
    if (g == 1) {
      std::vector<__nv_bfloat16> down_prompt;
      fill_prompt(kPrompt, y_cols, &down_prompt, 0xD040u);
      error = cudaMemcpy(workspace.prompt_projected_bf16_, down_prompt.data(),
                         down_prompt.size() * sizeof(down_prompt[0]),
                         cudaMemcpyHostToDevice);
      y_src = workspace.prompt_projected_bf16_;
    }
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_quantize_mmq_q8_1(
          qw38::cuda::QuantKind::kQ4K, y_src, kPrompt, y_cols,
          workspace.prompt_q8_, nullptr);
    }
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      return fail_cuda("screen quant", error);
    }
    float* out =
        g == 0 ? workspace.prompt_projection_a_ : workspace.prompt_mixer_output_;
    for (int t = 0; t < kTileCount; ++t) {
      float acc = 0.0F;
      int n = 0;
      for (int sample = 0; sample < 4; ++sample) {
        float ms = 0.0F;
        error = time_mmq(w, group.rows, group.columns, workspace.prompt_q8_,
                         kPrompt, out, kTiles[t], start, stop, &ms);
        if (error != cudaSuccess) {
          cudaEventDestroy(start);
          cudaEventDestroy(stop);
          return fail_cuda("screen time", error);
        }
        const qw38::cuda::MmqTileDispatch rec =
            qw38::cuda::last_mmq_tile_dispatch();
        if (sample == 0) {
          std::printf("screen_warmup group=%s tile=%s ms=%.9g pipeline=%s "
                      "kernel=%s\n",
                      group.id, kTiles[t].id, ms, json_bool(rec.pipeline),
                      rec.kernel);
        } else {
          acc += ms;
          ++n;
          std::printf("screen_sample group=%s tile=%s sample=%d ms=%.9g "
                      "pipeline=%s kernel=%s\n",
                      group.id, kTiles[t].id, sample, ms,
                      json_bool(rec.pipeline), rec.kernel);
        }
        if (!rec.pipeline || !rec.fma || !rec.async_y) {
          std::fprintf(stderr, "screen expected FMA/async Y for %s\n",
                       kTiles[t].id);
          cudaEventDestroy(start);
          cudaEventDestroy(stop);
          return 1;
        }
      }
      means[g][t] = n == 0 ? 0.0F : acc / static_cast<float>(n);
      std::printf("screen_mean group=%s tile=%s ms=%.9g\n", group.id,
                  kTiles[t].id, means[g][t]);
    }
  }

  int gate_winner = 0;
  int down_winner = 0;
  for (int t = 1; t < kTileCount; ++t) {
    if (means[0][t] < means[0][gate_winner]) gate_winner = t;
    if (means[1][t] < means[1][down_winner]) down_winner = t;
  }

  float control_ffn = 0.0F;
  float winner_ffn = 0.0F;
  auto time_ffn = [&](const Tile& gate, const Tile& down, float* mean) -> int {
    float acc = 0.0F;
    int n = 0;
    for (int sample = 0; sample < 4; ++sample) {
      float ms = 0.0F;
      error = cudaEventRecord(start, nullptr);
      if (error == cudaSuccess) {
        error = launch_complete_ffn(layer, &workspace, kPrompt, gate, down);
      }
      if (error == cudaSuccess) error = cudaEventRecord(stop, nullptr);
      if (error == cudaSuccess) error = cudaEventSynchronize(stop);
      if (error == cudaSuccess) error = cudaEventElapsedTime(&ms, start, stop);
      if (error != cudaSuccess) return fail_cuda("complete ffn", error);
      const qw38::cuda::MmqTileDispatch rec =
          qw38::cuda::last_mmq_tile_dispatch();
      if (sample == 0) {
        std::printf("complete_warmup gate=%s down=%s ms=%.9g last_kernel=%s "
                    "pipeline=%s\n",
                    gate.id, down.id, ms, rec.kernel, json_bool(rec.pipeline));
      } else {
        acc += ms;
        ++n;
        std::printf("complete_sample gate=%s down=%s sample=%d ms=%.9g "
                    "last_kernel=%s pipeline=%s\n",
                    gate.id, down.id, sample, ms, rec.kernel,
                    json_bool(rec.pipeline));
      }
    }
    *mean = n == 0 ? 0.0F : acc / static_cast<float>(n);
    return 0;
  };
  int rc = time_ffn(kTiles[kControl], kTiles[kControl], &control_ffn);
  if (rc == 0) {
    rc = time_ffn(kTiles[gate_winner], kTiles[down_winner], &winner_ffn);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (rc != 0) return rc;

  const bool keep = winner_ffn < control_ffn &&
                    (gate_winner != kControl || down_winner != kControl);
  const Tile& inst_gate = keep ? kTiles[gate_winner] : kTiles[kControl];
  const Tile& inst_down = keep ? kTiles[down_winner] : kTiles[kControl];
  std::printf(
      "screen_winner gate_up=%s down=%s control_ffn_ms=%.9g winner_ffn_ms=%.9g "
      "keep=%s installed_gate=%s installed_down=%s\n",
      kTiles[gate_winner].id, kTiles[down_winner].id, control_ffn, winner_ffn,
      json_bool(keep), inst_gate.id, inst_down.id);
  std::printf(
      "selected_ffn_gate=%u/%u up=%u/%u down=%u/%u pipeline=%s\n",
      qw38::cuda::selected_ffn_gate_quality_i(),
      qw38::cuda::selected_ffn_gate_prompt_tile(),
      qw38::cuda::selected_ffn_up_quality_i(),
      qw38::cuda::selected_ffn_up_prompt_tile(),
      qw38::cuda::selected_ffn_down_quality_i(),
      qw38::cuda::selected_ffn_down_prompt_tile(),
      qw38::cuda::selected_mmq_pipeline_path());
  return 0;
}

void emit_payload(const char* workload, const char* status) {
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-065\",\"workload\":\"%s\","
      "\"tier\":\"%s\",\"status\":\"%s\",\"selected_pipeline\":\"%s\","
      "\"gate_i\":%u,\"gate_j\":%u,\"down_i\":%u,\"down_j\":%u,"
      "\"claims_throughput\":false,\"llama_revision\":\"%s\","
      "\"gguf_sha256\":\"%s\"}\n",
      kPrefix, workload, qw38::cuda::test_tier_name(), status,
      qw38::cuda::selected_mmq_pipeline_path(),
      qw38::cuda::selected_ffn_gate_quality_i(),
      qw38::cuda::selected_ffn_gate_prompt_tile(),
      qw38::cuda::selected_ffn_down_quality_i(),
      qw38::cuda::selected_ffn_down_prompt_tile(), kLlamaRev, kGgufSha);
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
    rc = run_screen(options.model);
  } else {
    return usage(argv[0]);
  }
  emit_payload(workload, rc == 0 ? "passed" : "failed");
  std::printf("status=%s\n", rc == 0 ? "passed" : "failed");
  return rc;
}
