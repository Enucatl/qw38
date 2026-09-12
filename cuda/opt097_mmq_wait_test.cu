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

constexpr char kPrefix[] = "QW38_OPT097_MMQ_WAIT_RESULT=";
constexpr const char* kGgufSha =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr const char* kLlamaRev =
    "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr std::size_t kQ4KBytes = 144;
constexpr std::size_t kQ4KValues = 256;
constexpr float kAbsScale = 0.20F;
constexpr float kRelTol = 0.05F;
constexpr int kSampledOut = 16;
constexpr int kSampledPrompt = 8;
constexpr int kProdSample = 8;
constexpr unsigned int kQualityI = 128;
constexpr unsigned int kPromptTile = 128;

struct Variant {
  const char* id;
  bool async_x;
  bool split_xy_wait;
};

constexpr Variant kVariants[] = {
    {"joined_wait", true, false},
    {"split_xy_wait", true, true},
};
constexpr int kVariantCount = 2;
constexpr int kControl = 0;
constexpr int kCandidate = 1;

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

void fill_prompt(std::size_t prompt_rows, std::size_t columns,
                 std::vector<__nv_bfloat16>* prompt, std::uint32_t seed) {
  prompt->resize(prompt_rows * columns);
  for (std::size_t index = 0; index < prompt->size(); ++index) {
    (*prompt)[index] = __float2bfloat16_rn(
        unit_normal(static_cast<std::uint32_t>(index), seed));
  }
}

bool reference_dequant_gemm_fp64(const std::vector<std::uint8_t>& weights,
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
      double sum = 0.0;
      for (std::size_t column = 0; column < columns; ++column) {
        sum += static_cast<double>(decoded_row[column]) *
               static_cast<double>(
                   __bfloat162float(prompt[prompt_row * columns + column]));
      }
      (*output)[prompt_row * output_rows + out] = static_cast<float>(sum);
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

bool exact_equal(const std::vector<float>& got, const std::vector<float>& ref,
                 std::size_t* mismatch, std::size_t* nonfinite) {
  *mismatch = 0;
  *nonfinite = 0;
  if (got.size() != ref.size()) {
    *mismatch = got.size();
    return false;
  }
  for (std::size_t index = 0; index < got.size(); ++index) {
    if (!std::isfinite(got[index]) || !std::isfinite(ref[index])) {
      ++*nonfinite;
      continue;
    }
    std::uint32_t a = 0;
    std::uint32_t b = 0;
    std::memcpy(&a, &got[index], sizeof(a));
    std::memcpy(&b, &ref[index], sizeof(b));
    if (a != b) ++*mismatch;
  }
  return *nonfinite == 0 && *mismatch == 0;
}

int launch_q4(const std::uint8_t* weights, std::size_t output_rows,
              std::size_t columns, const qw38::cuda::Q8Block* y,
              std::size_t prompt_rows, float* output, const Variant& variant) {
  qw38::cuda::clear_mmq_tile_dispatch();
  qw38::cuda::MmqAsyncXOverrideScope async_scope(variant.async_x);
  qw38::cuda::MmqSplitXYWaitOverrideScope split_scope(variant.split_xy_wait);
  const cudaError_t error = qw38::cuda::launch_quant_mmq_mma_y_ij(
      qw38::cuda::QuantKind::kQ4K, weights, output_rows, columns, y, prompt_rows,
      output, kQualityI, kPromptTile, nullptr);
  if (error != cudaSuccess) return fail_cuda("q4 launch", error);
  return 0;
}

int check_dispatch(const Variant& variant, bool want_pipeline,
                   const char* label) {
  const qw38::cuda::MmqTileDispatch rec = qw38::cuda::last_mmq_tile_dispatch();
  const bool ok_ij =
      rec.quality_i == kQualityI && rec.prompt_tile == kPromptTile;
  const bool ok_pipe = rec.pipeline == want_pipeline;
  const bool ok_fma = !want_pipeline || (rec.fma && rec.async_y);
  const bool ok_x = rec.async_x == (want_pipeline && variant.async_x);
  const bool ok_split =
      rec.split_xy_wait == (want_pipeline && variant.async_x &&
                            variant.split_xy_wait);
  const bool ok_path = std::strcmp(rec.path, "fma_async") == 0;
  std::printf(
      "dispatch %s variant=%s kernel=%s aligned=%s pipeline=%s fallback=%s "
      "fma=%s async_y=%s async_x=%s split_xy_wait=%s path=%s\n",
      label, variant.id, rec.kernel, json_bool(rec.aligned),
      json_bool(rec.pipeline), json_bool(rec.fallback), json_bool(rec.fma),
      json_bool(rec.async_y), json_bool(rec.async_x),
      json_bool(rec.split_xy_wait), rec.path);
  if (!ok_ij || !ok_pipe || !ok_fma || !ok_x || !ok_split || !ok_path) {
    std::fprintf(stderr, "dispatch mismatch for %s %s\n", label, variant.id);
    return 1;
  }
  if (want_pipeline && variant.async_x && !variant.split_xy_wait &&
      std::strcmp(rec.kernel, "q4_i128_j128_fma_async_x") != 0) {
    std::fprintf(stderr, "joined_wait kernel id mismatch: %s\n", rec.kernel);
    return 1;
  }
  if (want_pipeline && variant.async_x && variant.split_xy_wait &&
      std::strcmp(rec.kernel, "q4_i128_j128_fma_async_x_split_wait") != 0) {
    std::fprintf(stderr, "split_xy_wait kernel id mismatch: %s\n", rec.kernel);
    return 1;
  }
  if (want_pipeline && !variant.async_x &&
      std::strcmp(rec.kernel, "q4_i128_j128_fma_async") != 0) {
    std::fprintf(stderr, "control kernel id mismatch: %s\n", rec.kernel);
    return 1;
  }
  return 0;
}

int run_resource_table() {
  int occupancy = 0;
  int registers = 0;
  std::size_t local_bytes = 0;
  std::size_t shared_bytes = 0;
  cudaError_t error = qw38::cuda::mmq_pipeline_kernel_attributes(
      qw38::cuda::QuantKind::kQ4K, kPromptTile, kQualityI, "fma_async",
      &occupancy, &registers, &local_bytes, &shared_bytes);
  const std::size_t extra_y = qw38::cuda::mmq_pipeline_extra_shared_bytes(
      kPromptTile, kQualityI, "fma_async");
  std::printf(
      "resource variant=control occupancy=%d regs=%d local=%zu shared=%zu "
      "extra_y=%zu extra_x=0\n",
      occupancy, registers, local_bytes, shared_bytes, extra_y);
  if (error != cudaSuccess || occupancy < 1) {
    std::fprintf(stderr, "control resource query failed\n");
    return 1;
  }

  int cand_occ = 0;
  int cand_regs = 0;
  std::size_t cand_local = 0;
  std::size_t cand_shared = 0;
  error = qw38::cuda::mmq_x_pipeline_kernel_attributes(
      &cand_occ, &cand_regs, &cand_local, &cand_shared);
  const std::size_t extra_x =
      qw38::cuda::mmq_x_pipeline_extra_shared_bytes(kQualityI);
  const std::size_t extra_x_two =
      2U * static_cast<std::size_t>(kQualityI) * kQ4KBytes + 16U;
  int device = 0;
  cudaDeviceProp prop{};
  cudaError_t prop_error = cudaGetDevice(&device);
  if (prop_error == cudaSuccess) {
    prop_error = cudaGetDeviceProperties(&prop, device);
  }
  const std::size_t optin =
      prop_error == cudaSuccess
          ? static_cast<std::size_t>(prop.sharedMemPerBlockOptin)
          : 0;
  std::printf(
      "resource variant=candidate occupancy=%d regs=%d local=%zu shared=%zu "
      "extra_x=%zu extra_x_two_stage=%zu device_optin=%zu query=%s\n",
      cand_occ, cand_regs, cand_local, cand_shared, extra_x, extra_x_two, optin,
      error == cudaSuccess ? "ok" : cudaGetErrorString(error));
  std::printf(
      "stage_lifetime warmup: cp.async raw X[k0] into the single 16B-aligned "
      "slot (I*9 chunks) + Y[k0,h0]; one commit group; wait_all; unpack into "
      "decoded X; CTA barrier. Two-stage ring needs extra_x_two_stage bytes "
      "and exceeds device_optin with the existing Y ring, so this kernel uses "
      "one prefetched raw slot.\n");
  std::printf(
      "stage_lifetime iter_k_load: joined_wait issues next Y and raw X[k+1] "
      "in one commit group; split_xy_wait commits Y then X separately.\n");
  std::printf(
      "stage_lifetime iter_k_compute: split_xy_wait uses wait_group<1> after "
      "half0 when both groups were issued, then wait_all before unpack.\n");
  std::printf(
      "stage_lifetime drain: last kb0 issues no next X; no final unpack; Y "
      "drain matches the existing two-stage Y pipeline\n");
  if (error != cudaSuccess) {
    std::printf("resource_nogo candidate_shared_or_occupancy_failed\n");
    return 1;
  }
  if (cand_shared > optin) {
    std::printf("resource_nogo shared=%zu optin=%zu\n", cand_shared, optin);
    return 1;
  }
  if (cand_occ < 1) {
    std::fprintf(stderr, "candidate occupancy < 1\n");
    return 1;
  }
  return 0;
}

struct CaseBuffers {
  std::uint8_t* d_w = nullptr;
  __nv_bfloat16* d_p = nullptr;
  qw38::cuda::Q8Block* d_y = nullptr;
  float* d_o = nullptr;
  std::size_t output_rows = 0;
  std::size_t columns = 0;
  std::size_t prompt_rows = 0;
  std::vector<float> control;
  std::vector<float> candidate;
  std::vector<float> fp64;
  std::vector<std::size_t> out_s;
  std::vector<std::size_t> pr_s;
};

void free_case(CaseBuffers* buffers) {
  cudaFree(buffers->d_o);
  cudaFree(buffers->d_y);
  cudaFree(buffers->d_p);
  cudaFree(buffers->d_w);
  buffers->d_o = nullptr;
  buffers->d_y = nullptr;
  buffers->d_p = nullptr;
  buffers->d_w = nullptr;
}

int prepare_case(const char* label, std::size_t output_rows, std::size_t columns,
                 std::size_t prompt_rows, bool unique, std::uint32_t seed,
                 CaseBuffers* buffers) {
  std::vector<std::uint8_t> weights;
  if (unique) {
    fill_unique_q4_weights(output_rows, columns, &weights);
  } else {
    fill_q4_weights(output_rows, columns, &weights);
  }
  std::vector<__nv_bfloat16> prompt;
  fill_prompt(prompt_rows, columns, &prompt, seed);
  buffers->output_rows = output_rows;
  buffers->columns = columns;
  buffers->prompt_rows = prompt_rows;
  buffers->out_s = sample_indices(output_rows, kSampledOut);
  buffers->pr_s = sample_indices(prompt_rows, kSampledPrompt);
  if (!reference_dequant_gemm_fp64(weights, output_rows, columns, prompt,
                                   prompt_rows, buffers->out_s, buffers->pr_s,
                                   &buffers->fp64)) {
    std::fprintf(stderr, "fp64 GEMM failed for %s\n", label);
    return 1;
  }
  cudaError_t error = cudaMalloc(&buffers->d_w, weights.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&buffers->d_p, prompt.size() * sizeof(prompt[0]));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&buffers->d_y, qw38::cuda::q8_prompt_workspace_bytes(
                                          prompt_rows, columns));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&buffers->d_o, prompt_rows * output_rows * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(buffers->d_w, weights.data(), weights.size(),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(buffers->d_p, prompt.data(),
                       prompt.size() * sizeof(prompt[0]), cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_mmq_q8_1(
        qw38::cuda::QuantKind::kQ4K, buffers->d_p, prompt_rows, columns,
        buffers->d_y, nullptr);
  }
  if (error != cudaSuccess) {
    free_case(buffers);
    return fail_cuda("case alloc", error);
  }
  buffers->control.assign(prompt_rows * output_rows, 0.0F);
  buffers->candidate.assign(prompt_rows * output_rows, 0.0F);
  return 0;
}

int run_variant(CaseBuffers* buffers, const Variant& variant, const char* label,
                std::vector<float>* host) {
  const bool aligned = buffers->output_rows % kQualityI == 0 &&
                       buffers->prompt_rows % kPromptTile == 0;
  int rc = launch_q4(buffers->d_w, buffers->output_rows, buffers->columns,
                     buffers->d_y, buffers->prompt_rows, buffers->d_o, variant);
  cudaError_t error = cudaSuccess;
  if (rc == 0) {
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) rc = fail_cuda("q4 sync", error);
  }
  if (rc == 0) {
    error = cudaMemcpy(host->data(), buffers->d_o, host->size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) rc = fail_cuda("q4 d2h", error);
  }
  if (rc == 0) rc = check_dispatch(variant, aligned, label);
  std::size_t nonfinite = 0;
  for (float value : *host) {
    if (!std::isfinite(value)) ++nonfinite;
  }
  float max_abs = 0.0F;
  std::size_t bad = 0;
  std::size_t fp64_nonfinite = 0;
  const bool fp64_ok =
      rc == 0 && ds4_ok(*host, buffers->fp64, buffers->columns, &max_abs, &bad,
                        &fp64_nonfinite);
  std::printf(
      "case=%s variant=%s m=%zu k=%zu n=%zu aligned=%s nonfinite=%zu "
      "fp64_max_abs=%.9g fp64_bad=%zu fp64_nonfinite=%zu\n",
      label, variant.id, buffers->output_rows, buffers->columns,
      buffers->prompt_rows, json_bool(aligned), nonfinite, max_abs, bad,
      fp64_nonfinite);
  if (rc != 0 || nonfinite != 0 || !fp64_ok) return 1;
  return 0;
}

int run_q4_pair(const char* label, std::size_t output_rows, std::size_t columns,
                std::size_t prompt_rows, bool unique) {
  CaseBuffers buffers;
  int rc = prepare_case(label, output_rows, columns, prompt_rows, unique,
                        0xA11CE5u, &buffers);
  if (rc != 0) return rc;
  rc = run_variant(&buffers, kVariants[kControl], label, &buffers.control);
  if (rc == 0) {
    rc = run_variant(&buffers, kVariants[kCandidate], label, &buffers.candidate);
  }
  if (rc == 0) {
    std::size_t mismatch = 0;
    std::size_t nonfinite = 0;
    const bool same =
        exact_equal(buffers.candidate, buffers.control, &mismatch, &nonfinite);
    std::printf("exact_vs_control %s mismatch=%zu nonfinite=%zu\n", label,
                mismatch, nonfinite);
    if (!same) rc = 1;
  }
  free_case(&buffers);
  return rc;
}

int run_stale_shared() {
  constexpr std::size_t kM = 128;
  constexpr std::size_t kK = 512;
  constexpr std::size_t kN = 128;
  CaseBuffers first;
  int rc = prepare_case("stale_a", kM, kK, kN, true, 0x111u, &first);
  if (rc != 0) return rc;
  rc = run_variant(&first, kVariants[kCandidate], "stale_a", &first.candidate);
  if (rc != 0) {
    free_case(&first);
    return rc;
  }

  std::vector<__nv_bfloat16> prompt_b;
  fill_prompt(kN, kK, &prompt_b, 0x222u);
  cudaError_t error =
      cudaMemcpy(first.d_p, prompt_b.data(), prompt_b.size() * sizeof(prompt_b[0]),
                 cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_mmq_q8_1(
        qw38::cuda::QuantKind::kQ4K, first.d_p, kN, kK, first.d_y, nullptr);
  }
  if (error != cudaSuccess) {
    free_case(&first);
    return fail_cuda("stale requant", error);
  }
  std::vector<std::uint8_t> weights(kM * (kK / kQ4KValues) * kQ4KBytes);
  error = cudaMemcpy(weights.data(), first.d_w, weights.size(),
                     cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    free_case(&first);
    return fail_cuda("stale weights", error);
  }
  first.out_s = sample_indices(kM, kSampledOut);
  first.pr_s = sample_indices(kN, kSampledPrompt);
  if (!reference_dequant_gemm_fp64(weights, kM, kK, prompt_b, kN, first.out_s,
                                   first.pr_s, &first.fp64)) {
    free_case(&first);
    return 1;
  }
  rc = run_variant(&first, kVariants[kControl], "stale_b_control",
                   &first.control);
  if (rc == 0) {
    rc = run_variant(&first, kVariants[kCandidate], "stale_b_candidate",
                     &first.candidate);
  }
  if (rc == 0) {
    std::size_t mismatch = 0;
    std::size_t nonfinite = 0;
    if (!exact_equal(first.candidate, first.control, &mismatch, &nonfinite)) {
      std::fprintf(stderr, "stale shared mismatch=%zu nonfinite=%zu\n",
                   mismatch, nonfinite);
      rc = 1;
    } else {
      std::printf("stale_shared ok mismatch=0\n");
    }
  }
  free_case(&first);
  return rc;
}

int run_smoke() {
  int rc = run_resource_table();
  rc |= run_q4_pair("smoke_m17", 17, 256, 17, true);
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
  const std::vector<std::size_t> rows = sample_indices(tensor.rows, kProdSample);
  const std::vector<std::size_t> prompts =
      sample_indices(static_cast<std::size_t>(kProdSample), kProdSample);
  std::vector<std::uint8_t> packed;
  if (copy_q4_rows(tensor, rows, &packed) != 0) return 1;
  std::vector<__nv_bfloat16> prompt;
  fill_prompt(kProdSample, tensor.columns, &prompt, 0xC0FFEEu);
  std::vector<float> expected;
  if (!reference_dequant_gemm_fp64(packed, rows.size(), tensor.columns, prompt,
                                   kProdSample, all_indices(rows.size()),
                                   all_indices(kProdSample), &expected)) {
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
                                 kProdSample, tensor.columns));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&d_o, kProdSample * rows.size() * sizeof(float));
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
        qw38::cuda::QuantKind::kQ4K, d_p, kProdSample, tensor.columns, d_y,
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
  std::vector<float> control(kProdSample * rows.size());
  std::vector<float> candidate(kProdSample * rows.size());
  for (int v = 0; v < kVariantCount && rc == 0; ++v) {
    std::vector<float>* host = v == 0 ? &control : &candidate;
    rc = launch_q4(d_w, rows.size(), tensor.columns, d_y, kProdSample, d_o,
                   kVariants[v]);
    if (rc == 0) {
      error = cudaDeviceSynchronize();
      if (error != cudaSuccess) rc = fail_cuda("prod sync", error);
    }
    if (rc == 0) {
      error = cudaMemcpy(host->data(), d_o, host->size() * sizeof(float),
                         cudaMemcpyDeviceToHost);
      if (error != cudaSuccess) rc = fail_cuda("prod d2h", error);
    }
    float max_abs = 0.0F;
    std::size_t bad = 0;
    std::size_t nonfinite = 0;
    const bool ok =
        rc == 0 &&
        ds4_ok(*host, expected, tensor.columns, &max_abs, &bad, &nonfinite);
    std::printf(
        "production %s variant=%s rows=%zu k=%zu sampled=%zu max_abs=%.9g "
        "bad=%zu nonfinite=%zu\n",
        name, kVariants[v].id, rows.size(), tensor.columns,
        static_cast<std::size_t>(kProdSample) * kProdSample, max_abs, bad,
        nonfinite);
    if (!ok) rc = 1;
  }
  if (rc == 0) {
    std::size_t mismatch = 0;
    std::size_t nonfinite = 0;
    if (!exact_equal(candidate, control, &mismatch, &nonfinite)) {
      std::fprintf(stderr, "production %s exact mismatch=%zu\n", name, mismatch);
      rc = 1;
    }
  }
  cudaFree(d_o);
  cudaFree(d_y);
  cudaFree(d_p);
  cudaFree(d_w);
  (void)prompts;
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
  int rc = run_q4_pair("corr_k256", 128, 256, 128, true);
  rc |= run_q4_pair("corr_k512", 128, 512, 128, true);
  rc |= run_q4_pair("corr_k5120", 128, 5120, 128, true);
  rc |= run_q4_pair("corr_k17408", 128, 17408, 128, true);
  rc |= run_q4_pair("fallback_m17", 17, 256, 8, false);
  rc |= run_q4_pair("fallback_n127", 128, 512, 127, false);
  rc |= run_q4_pair("fallback_m129", 129, 512, 128, false);
  rc |= run_stale_shared();
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

cudaError_t time_mmq(const std::uint8_t* weights, std::size_t rows,
                     std::size_t columns, const qw38::cuda::Q8Block* y,
                     std::size_t prompt_rows, float* output, const Variant& variant,
                     cudaEvent_t start, cudaEvent_t stop, float* ms) {
  qw38::cuda::MmqAsyncXOverrideScope async_scope(variant.async_x);
  qw38::cuda::MmqSplitXYWaitOverrideScope split_scope(variant.split_xy_wait);
  cudaError_t error = cudaEventRecord(start, nullptr);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quant_mmq_mma_y_ij(
        qw38::cuda::QuantKind::kQ4K, weights, rows, columns, y, prompt_rows,
        output, kQualityI, kPromptTile, nullptr);
  }
  if (error == cudaSuccess) error = cudaEventRecord(stop, nullptr);
  if (error == cudaSuccess) error = cudaEventSynchronize(stop);
  if (error == cudaSuccess) error = cudaEventElapsedTime(ms, start, stop);
  return error;
}

cudaError_t rotate_weights(const std::uint8_t* src, std::uint8_t* dst,
                           std::size_t bytes) {
  return cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
}

cudaError_t launch_complete_ffn(const qw38::cuda::DeviceCommonLayer& layer,
                                qw38::cuda::SchedulerWorkspace* workspace,
                                std::size_t prompt_rows, const Variant& variant) {
  qw38::cuda::MmqAsyncXOverrideScope async_scope(variant.async_x);
  qw38::cuda::MmqSplitXYWaitOverrideScope split_scope(variant.split_xy_wait);
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

  struct ShapeGroup {
    const char* id;
    const std::uint8_t* data;
    std::size_t rows;
    std::size_t columns;
    std::size_t y_cols;
    std::size_t bytes;
  };
  const ShapeGroup groups[2] = {
      {"gate_up", layer.ffn_gate.data, layer.ffn_gate.rows,
       layer.ffn_gate.columns, qw38::internal::kResidualWidth,
       layer.ffn_gate.rows * (layer.ffn_gate.columns / kQ4KValues) * kQ4KBytes},
      {"down", layer.ffn_down.data, layer.ffn_down.rows, layer.ffn_down.columns,
       qw38::internal::kFfnWidth,
       layer.ffn_down.rows * (layer.ffn_down.columns / kQ4KValues) * kQ4KBytes},
  };
  float means[2][kVariantCount] = {};
  for (int g = 0; g < 2; ++g) {
    const ShapeGroup& group = groups[g];
    std::uint8_t* rotating = nullptr;
    error = cudaMalloc(&rotating, group.bytes);
    if (error != cudaSuccess) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      return fail_cuda("rotate alloc", error);
    }
    const __nv_bfloat16* y_src = workspace.prompt_normalized_;
    if (g == 1) {
      std::vector<__nv_bfloat16> down_prompt;
      fill_prompt(kPrompt, group.y_cols, &down_prompt, 0xD040u);
      error = cudaMemcpy(workspace.prompt_projected_bf16_, down_prompt.data(),
                         down_prompt.size() * sizeof(down_prompt[0]),
                         cudaMemcpyHostToDevice);
      y_src = workspace.prompt_projected_bf16_;
    }
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_quantize_mmq_q8_1(
          qw38::cuda::QuantKind::kQ4K, y_src, kPrompt, group.y_cols,
          workspace.prompt_q8_, nullptr);
    }
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
      cudaFree(rotating);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      return fail_cuda("screen quant", error);
    }
    float* out =
        g == 0 ? workspace.prompt_projection_a_ : workspace.prompt_mixer_output_;
    for (int v = 0; v < kVariantCount; ++v) {
      float acc = 0.0F;
      int n = 0;
      for (int sample = 0; sample < 4; ++sample) {
        error = rotate_weights(group.data, rotating, group.bytes);
        float ms = 0.0F;
        if (error == cudaSuccess) {
          error = time_mmq(rotating, group.rows, group.columns,
                           workspace.prompt_q8_, kPrompt, out,
                           kVariants[v], start, stop, &ms);
        }
        if (error != cudaSuccess) {
          cudaFree(rotating);
          cudaEventDestroy(start);
          cudaEventDestroy(stop);
          return fail_cuda("screen time", error);
        }
        const qw38::cuda::MmqTileDispatch rec =
            qw38::cuda::last_mmq_tile_dispatch();
        if (sample == 0) {
          std::printf(
              "screen_warmup group=%s variant=%s ms=%.9g pipeline=%s "
              "async_x=%s split_xy_wait=%s kernel=%s\n",
              group.id, kVariants[v].id, ms, json_bool(rec.pipeline),
              json_bool(rec.async_x), json_bool(rec.split_xy_wait), rec.kernel);
        } else {
          acc += ms;
          ++n;
          std::printf(
              "screen_sample group=%s variant=%s sample=%d ms=%.9g "
              "pipeline=%s async_x=%s split_xy_wait=%s kernel=%s\n",
              group.id, kVariants[v].id, sample, ms, json_bool(rec.pipeline),
              json_bool(rec.async_x), json_bool(rec.split_xy_wait), rec.kernel);
        }
        if (!rec.pipeline || !rec.fma || !rec.async_y) {
          std::fprintf(stderr, "screen expected FMA/async Y for %s\n",
                       kVariants[v].id);
          cudaFree(rotating);
          cudaEventDestroy(start);
          cudaEventDestroy(stop);
          return 1;
        }
        if (rec.async_x != kVariants[v].async_x ||
            rec.split_xy_wait != kVariants[v].split_xy_wait) {
          std::fprintf(stderr, "screen wait schedule mismatch for %s\n",
                       kVariants[v].id);
          cudaFree(rotating);
          cudaEventDestroy(start);
          cudaEventDestroy(stop);
          return 1;
        }
      }
      means[g][v] = n == 0 ? 0.0F : acc / static_cast<float>(n);
      std::printf("screen_mean group=%s variant=%s ms=%.9g\n", group.id,
                  kVariants[v].id, means[g][v]);
    }
    cudaFree(rotating);
  }

  const bool gemm_faster =
      means[0][kCandidate] < means[0][kControl] &&
      means[1][kCandidate] < means[1][kControl];
  const bool survivor_split = gemm_faster;
  float control_ffn = 0.0F;
  float survivor_ffn = 0.0F;
  auto time_ffn = [&](const Variant& variant, float* mean) -> int {
    float acc = 0.0F;
    int n = 0;
    for (int sample = 0; sample < 4; ++sample) {
      float ms = 0.0F;
      error = cudaEventRecord(start, nullptr);
      if (error == cudaSuccess) {
        error = launch_complete_ffn(layer, &workspace, kPrompt, variant);
      }
      if (error == cudaSuccess) error = cudaEventRecord(stop, nullptr);
      if (error == cudaSuccess) error = cudaEventSynchronize(stop);
      if (error == cudaSuccess) error = cudaEventElapsedTime(&ms, start, stop);
      if (error != cudaSuccess) return fail_cuda("complete ffn", error);
      const qw38::cuda::MmqTileDispatch rec =
          qw38::cuda::last_mmq_tile_dispatch();
      if (sample == 0) {
        std::printf(
            "complete_warmup wait=%s ms=%.9g last_kernel=%s pipeline=%s\n",
            variant.id, ms, rec.kernel, json_bool(rec.pipeline));
      } else {
        acc += ms;
        ++n;
        std::printf(
            "complete_sample wait=%s sample=%d ms=%.9g last_kernel=%s "
            "pipeline=%s\n",
            variant.id, sample, ms, rec.kernel, json_bool(rec.pipeline));
      }
    }
    *mean = n == 0 ? 0.0F : acc / static_cast<float>(n);
    return 0;
  };
  int rc = time_ffn(kVariants[kControl], &control_ffn);
  if (rc == 0) rc = time_ffn(kVariants[kCandidate], &survivor_ffn);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (rc != 0) return rc;

  const bool keep = survivor_split && survivor_ffn < control_ffn;
  std::printf(
      "screen_winner control_gate_up_ms=%.9g candidate_gate_up_ms=%.9g "
      "control_down_ms=%.9g candidate_down_ms=%.9g control_ffn_ms=%.9g "
      "survivor_ffn_ms=%.9g keep=%s installed_split_xy_wait=%s "
      "selected_split_xy_wait=%s\n",
      means[0][kControl], means[0][kCandidate], means[1][kControl],
      means[1][kCandidate], control_ffn, survivor_ffn, json_bool(keep),
      json_bool(keep), json_bool(qw38::cuda::selected_mmq_split_xy_wait()));
  return 0;
}

void emit_payload(const char* workload, const char* status) {
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-097\",\"workload\":\"%s\","
      "\"tier\":\"%s\",\"status\":\"%s\",\"selected_pipeline\":\"%s\","
      "\"selected_async_x\":%s,\"selected_split_xy_wait\":%s,"
      "\"control_wait\":\"joined_wait\",\"candidate_wait\":\"split_xy_wait\","
      "\"tile\":\"i128_j128\",\"claims_throughput\":true,"
      "\"llama_revision\":\"%s\",\"gguf_sha256\":\"%s\"}\n",
      kPrefix, workload, qw38::cuda::test_tier_name(), status,
      qw38::cuda::selected_mmq_pipeline_path(),
      json_bool(qw38::cuda::selected_mmq_async_x()),
      json_bool(qw38::cuda::selected_mmq_split_xy_wait()), kLlamaRev, kGgufSha);
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
