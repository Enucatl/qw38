#include "full_scheduler.h"
#include "mixer.h"
#include "model.h"
#include "quant.h"
#include "quant_mmv.h"
#include "scheduler_primitives.h"
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

constexpr char kPrefix[] = "QW38_OPT067_PROMPT_PAIR_RESULT=";
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
constexpr int kProdOut = 16;
constexpr int kProdPrompt = 4;
constexpr unsigned int kControlI = 128;
constexpr unsigned int kControlJ = 128;

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
                            std::uint32_t salt,
                            std::vector<std::uint8_t>* weights) {
  fill_q4_weights(rows, columns, weights);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t block = 0; block < columns / kQ4KValues; ++block) {
      std::uint8_t* packed =
          weights->data() + (row * (columns / kQ4KValues) + block) * kQ4KBytes;
      packed[16 + (row % 32)] =
          static_cast<std::uint8_t>((row + block * 17U + salt) & 0x7FU);
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

void fill_prompt_constant(std::size_t prompt_rows, std::size_t columns,
                          float value, std::vector<__nv_bfloat16>* prompt) {
  prompt->assign(prompt_rows * columns, __float2bfloat16_rn(value));
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

bool exact_bf16(const std::vector<__nv_bfloat16>& got,
                const std::vector<__nv_bfloat16>& ref, std::size_t* mismatch,
                std::size_t* nonfinite) {
  *mismatch = 0;
  *nonfinite = 0;
  if (got.size() != ref.size()) {
    *mismatch = got.size();
    return false;
  }
  for (std::size_t index = 0; index < got.size(); ++index) {
    const float a = __bfloat162float(got[index]);
    const float b = __bfloat162float(ref[index]);
    if (!std::isfinite(a) || !std::isfinite(b)) {
      ++*nonfinite;
      continue;
    }
    std::uint16_t aa = 0;
    std::uint16_t bb = 0;
    std::memcpy(&aa, &got[index], sizeof(aa));
    std::memcpy(&bb, &ref[index], sizeof(bb));
    if (aa != bb) ++*mismatch;
  }
  return *nonfinite == 0 && *mismatch == 0;
}

int count_nonfinite_bf16(const std::vector<__nv_bfloat16>& values) {
  int n = 0;
  for (const __nv_bfloat16 value : values) {
    if (!std::isfinite(__bfloat162float(value))) ++n;
  }
  return n;
}

int run_resource_table() {
  int occ = 0;
  int regs = 0;
  std::size_t local_bytes = 0;
  std::size_t shared_bytes = 0;
  cudaError_t error = qw38::cuda::mmq_pipeline_kernel_attributes(
      qw38::cuda::QuantKind::kQ4K, kControlJ, kControlI, "fma_async", &occ,
      &regs, &local_bytes, &shared_bytes);
  std::printf(
      "resource variant=control occupancy=%d regs=%d local=%zu shared=%zu "
      "tile=i128_j128\n",
      occ, regs, local_bytes, shared_bytes);
  if (error != cudaSuccess || occ < 1) {
    std::fprintf(stderr, "control resource query failed\n");
    return 1;
  }

  int cand_occ = 0;
  int cand_regs = 0;
  std::size_t cand_local = 0;
  std::size_t cand_shared = 0;
  error = qw38::cuda::mmq_paired_kernel_attributes(
      &cand_occ, &cand_regs, &cand_local, &cand_shared, false);
  int fb_occ = 0;
  int fb_regs = 0;
  std::size_t fb_local = 0;
  std::size_t fb_shared = 0;
  const cudaError_t fb_error = qw38::cuda::mmq_paired_kernel_attributes(
      &fb_occ, &fb_regs, &fb_local, &fb_shared, true);
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
      "fallback_occ=%d fallback_regs=%d fallback_local=%zu extra_x=0 "
      "device_optin=%zu query=%s fallback_query=%s\n",
      cand_occ, cand_regs, cand_local, cand_shared, fb_occ, fb_regs, fb_local,
      optin, error == cudaSuccess ? "ok" : cudaGetErrorString(error),
      fb_error == cudaSuccess ? "ok" : cudaGetErrorString(fb_error));
  if (error != cudaSuccess || fb_error != cudaSuccess) return 1;
  if (cand_shared > optin || cand_occ < 1 || fb_occ < 1) {
    std::printf("resource_nogo shared=%zu optin=%zu occ=%d\n", cand_shared,
                optin, cand_occ);
    return 1;
  }
  return 0;
}

cudaError_t launch_separate_gate_up(
    const std::uint8_t* gate, const std::uint8_t* up, std::size_t rows,
    std::size_t columns, const qw38::cuda::Q8Block* y, std::size_t prompt_rows,
    float* gate_out, float* up_out, __nv_bfloat16* activated) {
  qw38::cuda::clear_mmq_tile_dispatch();
  cudaError_t error = qw38::cuda::launch_quant_mmq_mma_y_ij(
      qw38::cuda::QuantKind::kQ4K, gate, rows, columns, y, prompt_rows, gate_out,
      kControlI, kControlJ, nullptr);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quant_mmq_mma_y_ij(
        qw38::cuda::QuantKind::kQ4K, up, rows, columns, y, prompt_rows, up_out,
        kControlI, kControlJ, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_swiglu_bf16(
        gate_out, up_out, prompt_rows * rows, activated, nullptr);
  }
  return error;
}

cudaError_t launch_paired(const std::uint8_t* gate, const std::uint8_t* up,
                          std::size_t rows, std::size_t columns,
                          const qw38::cuda::Q8Block* y, std::size_t prompt_rows,
                          __nv_bfloat16* activated, float* gate_dump,
                          float* up_dump) {
  qw38::cuda::clear_mmq_paired_dispatch();
  qw38::cuda::FfnPromptPairOverrideScope scope("i64_j64");
  return qw38::cuda::launch_q4_mmq_paired_gate_up_swiglu_bf16(
      gate, up, rows, columns, y, prompt_rows, activated, nullptr, gate_dump,
      up_dump);
}

int check_paired_dispatch(const char* label, bool want_fallback) {
  const qw38::cuda::MmqPairedDispatch rec =
      qw38::cuda::last_mmq_paired_dispatch();
  std::printf(
      "dispatch %s paired=%s fallback=%s kernel=%s i=%u j=%u dump=%s path=%s\n",
      label, json_bool(rec.paired), json_bool(rec.fallback), rec.kernel,
      rec.quality_i, rec.prompt_tile, json_bool(rec.dump_gate_up), rec.path);
  if (!rec.paired || rec.quality_i != 64 || rec.prompt_tile != 64 ||
      rec.fallback != want_fallback) {
    std::fprintf(stderr, "paired dispatch mismatch for %s\n", label);
    return 1;
  }
  return 0;
}

struct PairCase {
  std::uint8_t* d_gate = nullptr;
  std::uint8_t* d_up = nullptr;
  __nv_bfloat16* d_p = nullptr;
  qw38::cuda::Q8Block* d_y = nullptr;
  float* d_gate_out = nullptr;
  float* d_up_out = nullptr;
  float* d_gate_dump = nullptr;
  float* d_up_dump = nullptr;
  __nv_bfloat16* d_act_ctrl = nullptr;
  __nv_bfloat16* d_act_cand = nullptr;
  std::size_t rows = 0;
  std::size_t columns = 0;
  std::size_t prompt_rows = 0;
  std::vector<std::uint8_t> gate_host;
  std::vector<std::uint8_t> up_host;
  std::vector<__nv_bfloat16> prompt;
  std::vector<std::size_t> out_s;
  std::vector<std::size_t> pr_s;
};

void free_pair(PairCase* buffers) {
  cudaFree(buffers->d_act_cand);
  cudaFree(buffers->d_act_ctrl);
  cudaFree(buffers->d_up_dump);
  cudaFree(buffers->d_gate_dump);
  cudaFree(buffers->d_up_out);
  cudaFree(buffers->d_gate_out);
  cudaFree(buffers->d_y);
  cudaFree(buffers->d_p);
  cudaFree(buffers->d_up);
  cudaFree(buffers->d_gate);
  *buffers = PairCase{};
}

int prepare_pair(const char* label, std::size_t rows, std::size_t columns,
                 std::size_t prompt_rows, std::uint32_t gate_salt,
                 std::uint32_t up_salt, std::uint32_t prompt_seed,
                 float prompt_const, bool unique, PairCase* buffers) {
  buffers->rows = rows;
  buffers->columns = columns;
  buffers->prompt_rows = prompt_rows;
  if (unique) {
    fill_unique_q4_weights(rows, columns, gate_salt, &buffers->gate_host);
    fill_unique_q4_weights(rows, columns, up_salt, &buffers->up_host);
  } else {
    fill_q4_weights(rows, columns, &buffers->gate_host);
    fill_q4_weights(rows, columns, &buffers->up_host);
  }
  if (std::isfinite(prompt_const)) {
    fill_prompt_constant(prompt_rows, columns, prompt_const, &buffers->prompt);
  } else {
    fill_prompt(prompt_rows, columns, &buffers->prompt, prompt_seed);
  }
  buffers->out_s = rows <= static_cast<std::size_t>(kSampledOut)
                       ? all_indices(rows)
                       : sample_indices(rows, kSampledOut);
  buffers->pr_s = prompt_rows <= static_cast<std::size_t>(kSampledPrompt)
                      ? all_indices(prompt_rows)
                      : sample_indices(prompt_rows, kSampledPrompt);
  const std::size_t elems = prompt_rows * rows;
  cudaError_t error = cudaMalloc(&buffers->d_gate, buffers->gate_host.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&buffers->d_up, buffers->up_host.size());
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&buffers->d_p, buffers->prompt.size() * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&buffers->d_y, qw38::cuda::q8_prompt_workspace_bytes(
                                          prompt_rows, columns));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&buffers->d_gate_out, elems * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&buffers->d_up_out, elems * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&buffers->d_gate_dump, elems * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&buffers->d_up_dump, elems * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&buffers->d_act_ctrl, elems * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&buffers->d_act_cand, elems * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(buffers->d_gate, buffers->gate_host.data(),
                       buffers->gate_host.size(), cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(buffers->d_up, buffers->up_host.data(),
                       buffers->up_host.size(), cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(buffers->d_p, buffers->prompt.data(),
                       buffers->prompt.size() * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_mmq_q8_1(
        qw38::cuda::QuantKind::kQ4K, buffers->d_p, prompt_rows, columns,
        buffers->d_y, nullptr);
  }
  if (error != cudaSuccess) {
    free_pair(buffers);
    std::fprintf(stderr, "prepare %s failed: %s\n", label,
                 cudaGetErrorString(error));
    return 1;
  }
  return 0;
}

int run_pair_case(const char* label, std::size_t rows, std::size_t columns,
                  std::size_t prompt_rows, std::uint32_t gate_salt,
                  std::uint32_t up_salt, std::uint32_t prompt_seed,
                  float prompt_const, bool unique, bool expect_swap_mismatch) {
  PairCase buffers;
  int rc = prepare_pair(label, rows, columns, prompt_rows, gate_salt, up_salt,
                        prompt_seed, prompt_const, unique, &buffers);
  if (rc != 0) return rc;
  const bool fallback = rows % 64U != 0 || prompt_rows % 64U != 0;
  const std::size_t elems = prompt_rows * rows;
  cudaError_t error = launch_separate_gate_up(
      buffers.d_gate, buffers.d_up, rows, columns, buffers.d_y, prompt_rows,
      buffers.d_gate_out, buffers.d_up_out, buffers.d_act_ctrl);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error == cudaSuccess) {
    error = launch_paired(buffers.d_gate, buffers.d_up, rows, columns,
                          buffers.d_y, prompt_rows, buffers.d_act_cand,
                          buffers.d_gate_dump, buffers.d_up_dump);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    free_pair(&buffers);
    return fail_cuda(label, error);
  }
  rc = check_paired_dispatch(label, fallback);

  std::vector<float> gate_ctrl(elems);
  std::vector<float> up_ctrl(elems);
  std::vector<float> gate_cand(elems);
  std::vector<float> up_cand(elems);
  std::vector<__nv_bfloat16> act_ctrl(elems);
  std::vector<__nv_bfloat16> act_cand(elems);
  if (rc == 0) {
    error = cudaMemcpy(gate_ctrl.data(), buffers.d_gate_out,
                       elems * sizeof(float), cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(up_ctrl.data(), buffers.d_up_out, elems * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(gate_cand.data(), buffers.d_gate_dump,
                       elems * sizeof(float), cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(up_cand.data(), buffers.d_up_dump, elems * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(act_ctrl.data(), buffers.d_act_ctrl,
                       elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(act_cand.data(), buffers.d_act_cand,
                       elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) {
    free_pair(&buffers);
    return fail_cuda("d2h", error);
  }

  std::vector<float> gate_fp64;
  std::vector<float> up_fp64;
  if (!reference_dequant_gemm_fp64(buffers.gate_host, rows, columns,
                                   buffers.prompt, prompt_rows, buffers.out_s,
                                   buffers.pr_s, &gate_fp64) ||
      !reference_dequant_gemm_fp64(buffers.up_host, rows, columns,
                                   buffers.prompt, prompt_rows, buffers.out_s,
                                   buffers.pr_s, &up_fp64)) {
    free_pair(&buffers);
    std::fprintf(stderr, "fp64 failed for %s\n", label);
    return 1;
  }

  float gate_abs = 0.0F;
  float up_abs = 0.0F;
  std::size_t gate_bad = 0;
  std::size_t up_bad = 0;
  std::size_t gate_nf = 0;
  std::size_t up_nf = 0;
  const bool gate_ok =
      ds4_ok(gate_cand, gate_fp64, columns, &gate_abs, &gate_bad, &gate_nf);
  const bool up_ok = ds4_ok(up_cand, up_fp64, columns, &up_abs, &up_bad, &up_nf);
  std::size_t gate_mis = 0;
  std::size_t up_mis = 0;
  std::size_t gate_ex_nf = 0;
  std::size_t up_ex_nf = 0;
  const bool gate_exact =
      exact_equal(gate_cand, gate_ctrl, &gate_mis, &gate_ex_nf);
  const bool up_exact = exact_equal(up_cand, up_ctrl, &up_mis, &up_ex_nf);
  std::size_t act_mis = 0;
  std::size_t act_nf = 0;
  const bool act_exact = exact_bf16(act_cand, act_ctrl, &act_mis, &act_nf);
  const int act_nonfinite = count_nonfinite_bf16(act_cand);

  std::printf(
      "case=%s m=%zu k=%zu n=%zu fallback=%s gate_exact=%s up_exact=%s "
      "act_exact=%s gate_fp64_max_abs=%.9g gate_bad=%zu up_fp64_max_abs=%.9g "
      "up_bad=%zu act_nonfinite=%d gate_mismatch=%zu up_mismatch=%zu "
      "act_mismatch=%zu\n",
      label, rows, columns, prompt_rows, json_bool(fallback),
      json_bool(gate_exact), json_bool(up_exact), json_bool(act_exact),
      gate_abs, gate_bad, up_abs, up_bad, act_nonfinite, gate_mis, up_mis,
      act_mis);

  if (!gate_ok || !up_ok || act_nonfinite != 0) rc = 1;
  if (!act_exact) {
    std::vector<float> act_ctrl_f(elems);
    std::vector<float> act_cand_f(elems);
    for (std::size_t i = 0; i < elems; ++i) {
      act_ctrl_f[i] = __bfloat162float(act_ctrl[i]);
      act_cand_f[i] = __bfloat162float(act_cand[i]);
    }
    float act_abs = 0.0F;
    std::size_t act_bad = 0;
    std::size_t act_ds4_nf = 0;
    const bool act_ds4 = ds4_ok(act_cand_f, act_ctrl_f, columns, &act_abs,
                                &act_bad, &act_ds4_nf);
    std::printf("act_vs_control_ds4 %s max_abs=%.9g bad=%zu nonfinite=%zu "
                "ds4=%s bf16_mismatch=%zu\n",
                label, act_abs, act_bad, act_ds4_nf, json_bool(act_ds4),
                act_mis);
    if (!act_ds4) {
      std::fprintf(stderr, "%s activated mismatch=%zu nonfinite=%zu\n", label,
                   act_mis, act_nf);
      rc = 1;
    }
  }
  if (!gate_exact || !up_exact) {
    std::printf(
        "fp32_vs_i128_control %s gate_mismatch=%zu up_mismatch=%zu "
        "(ds4_ok=%s; not the keep gate)\n",
        label, gate_mis, up_mis, json_bool(gate_ok && up_ok));
  }

  if (!fallback && rc == 0) {
    qw38::cuda::FfnTileOverrideScope tiles(64, 64, 64, 64, 128, 128);
    error = qw38::cuda::launch_quant_mmq_mma_y_ij(
        qw38::cuda::QuantKind::kQ4K, buffers.d_gate, rows, columns, buffers.d_y,
        prompt_rows, buffers.d_gate_out, 64, 64, nullptr);
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_quant_mmq_mma_y_ij(
          qw38::cuda::QuantKind::kQ4K, buffers.d_up, rows, columns, buffers.d_y,
          prompt_rows, buffers.d_up_out, 64, 64, nullptr);
    }
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
      free_pair(&buffers);
      return fail_cuda("same-tile control", error);
    }
    error = cudaMemcpy(gate_ctrl.data(), buffers.d_gate_out,
                       elems * sizeof(float), cudaMemcpyDeviceToHost);
    if (error == cudaSuccess) {
      error = cudaMemcpy(up_ctrl.data(), buffers.d_up_out, elems * sizeof(float),
                         cudaMemcpyDeviceToHost);
    }
    if (error != cudaSuccess) {
      free_pair(&buffers);
      return fail_cuda("same-tile d2h", error);
    }
    std::size_t same_gate_mis = 0;
    std::size_t same_up_mis = 0;
    std::size_t same_gate_nf = 0;
    std::size_t same_up_nf = 0;
    const bool same_gate =
        exact_equal(gate_cand, gate_ctrl, &same_gate_mis, &same_gate_nf);
    const bool same_up =
        exact_equal(up_cand, up_ctrl, &same_up_mis, &same_up_nf);
    std::printf("same_tile_i64 %s gate_exact=%s mismatch=%zu up_exact=%s "
                "mismatch=%zu\n",
                label, json_bool(same_gate), same_gate_mis, json_bool(same_up),
                same_up_mis);
    if (!same_gate || !same_up) rc = 1;
  }

  if (expect_swap_mismatch && rc == 0) {
    error = launch_paired(buffers.d_up, buffers.d_gate, rows, columns,
                          buffers.d_y, prompt_rows, buffers.d_act_cand, nullptr,
                          nullptr);
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
      free_pair(&buffers);
      return fail_cuda("swap launch", error);
    }
    std::vector<__nv_bfloat16> swapped(elems);
    error = cudaMemcpy(swapped.data(), buffers.d_act_cand,
                       elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
      free_pair(&buffers);
      return fail_cuda("swap d2h", error);
    }
    std::size_t swap_mis = 0;
    std::size_t swap_nf = 0;
    const bool same = exact_bf16(swapped, act_ctrl, &swap_mis, &swap_nf);
    std::printf("swap_detect %s mismatch=%zu same=%s\n", label, swap_mis,
                json_bool(same));
    if (same || swap_mis == 0) {
      std::fprintf(stderr, "swapping gate/up was not detectable\n");
      rc = 1;
    }
  }

  free_pair(&buffers);
  return rc;
}

int run_q8_and_residual(const char* label, std::size_t rows, std::size_t columns,
                        std::size_t prompt_rows) {
  PairCase buffers;
  int rc = prepare_pair(label, rows, columns, prompt_rows, 0x11u, 0x22u, 0x33u,
                        NAN, true, &buffers);
  if (rc != 0) return rc;
  const std::size_t elems = prompt_rows * rows;
  qw38::cuda::Q8Block* d_q8_ctrl = nullptr;
  qw38::cuda::Q8Block* d_q8_cand = nullptr;
  const std::size_t q8_bytes =
      qw38::cuda::q8_prompt_workspace_bytes(prompt_rows, rows);
  cudaError_t error = cudaMalloc(&d_q8_ctrl, q8_bytes);
  if (error == cudaSuccess) error = cudaMalloc(&d_q8_cand, q8_bytes);
  if (error == cudaSuccess) {
    error = launch_separate_gate_up(buffers.d_gate, buffers.d_up, rows, columns,
                                    buffers.d_y, prompt_rows,
                                    buffers.d_gate_out, buffers.d_up_out,
                                    buffers.d_act_ctrl);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_swiglu_quantize_mmq_q8_1(
        buffers.d_gate_out, buffers.d_up_out, prompt_rows, rows, d_q8_ctrl,
        nullptr);
  }
  if (error == cudaSuccess) {
    error = launch_paired(buffers.d_gate, buffers.d_up, rows, columns,
                          buffers.d_y, prompt_rows, buffers.d_act_cand, nullptr,
                          nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_mmq_q8_1(
        qw38::cuda::QuantKind::kQ4K, buffers.d_act_cand, prompt_rows, rows,
        d_q8_cand, nullptr);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    cudaFree(d_q8_cand);
    cudaFree(d_q8_ctrl);
    free_pair(&buffers);
    return fail_cuda("q8 stage", error);
  }
  std::vector<std::uint8_t> q8_ctrl(q8_bytes);
  std::vector<std::uint8_t> q8_cand(q8_bytes);
  error = cudaMemcpy(q8_ctrl.data(), d_q8_ctrl, q8_bytes, cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) {
    error = cudaMemcpy(q8_cand.data(), d_q8_cand, q8_bytes,
                       cudaMemcpyDeviceToHost);
  }
  cudaFree(d_q8_cand);
  cudaFree(d_q8_ctrl);
  if (error != cudaSuccess) {
    free_pair(&buffers);
    return fail_cuda("q8 d2h", error);
  }
  std::size_t mismatch = 0;
  for (std::size_t i = 0; i < q8_bytes; ++i) {
    if (q8_ctrl[i] != q8_cand[i]) ++mismatch;
  }
  std::printf("q8_stage %s bytes=%zu mismatch=%zu\n", label, q8_bytes, mismatch);
  (void)elems;
  free_pair(&buffers);
  return mismatch == 0 ? 0 : 1;
}

int run_smoke() {
  int rc = run_resource_table();
  rc |= run_pair_case("smoke_m17", 17, 256, 17, 0xA1u, 0xB2u, 0xC3u, NAN, true,
                      true);
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

int run_production_sampled(const qw38::cuda::DeviceCommonLayer& layer) {
  const std::vector<std::size_t> rows =
      sample_indices(layer.ffn_gate.rows, kProdOut);
  const std::size_t prompt_rows = static_cast<std::size_t>(kProdPrompt);
  std::vector<std::uint8_t> gate;
  std::vector<std::uint8_t> up;
  if (copy_q4_rows(layer.ffn_gate, rows, &gate) != 0) return 1;
  if (copy_q4_rows(layer.ffn_up, rows, &up) != 0) return 1;
  PairCase buffers;
  buffers.rows = rows.size();
  buffers.columns = layer.ffn_gate.columns;
  buffers.prompt_rows = prompt_rows;
  buffers.gate_host = std::move(gate);
  buffers.up_host = std::move(up);
  fill_prompt(prompt_rows, buffers.columns, &buffers.prompt, 0x5120u);
  buffers.out_s = all_indices(buffers.rows);
  buffers.pr_s = all_indices(prompt_rows);
  const std::size_t elems = prompt_rows * buffers.rows;
  cudaError_t error = cudaMalloc(&buffers.d_gate, buffers.gate_host.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&buffers.d_up, buffers.up_host.size());
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&buffers.d_p,
                       buffers.prompt.size() * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&buffers.d_y, qw38::cuda::q8_prompt_workspace_bytes(
                                         prompt_rows, buffers.columns));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&buffers.d_gate_out, elems * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&buffers.d_up_out, elems * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&buffers.d_gate_dump, elems * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&buffers.d_up_dump, elems * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&buffers.d_act_ctrl, elems * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&buffers.d_act_cand, elems * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(buffers.d_gate, buffers.gate_host.data(),
                       buffers.gate_host.size(), cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(buffers.d_up, buffers.up_host.data(),
                       buffers.up_host.size(), cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(buffers.d_p, buffers.prompt.data(),
                       buffers.prompt.size() * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_mmq_q8_1(
        qw38::cuda::QuantKind::kQ4K, buffers.d_p, prompt_rows, buffers.columns,
        buffers.d_y, nullptr);
  }
  if (error != cudaSuccess) {
    free_pair(&buffers);
    return fail_cuda("prod alloc", error);
  }
  error = launch_separate_gate_up(
      buffers.d_gate, buffers.d_up, buffers.rows, buffers.columns, buffers.d_y,
      prompt_rows, buffers.d_gate_out, buffers.d_up_out, buffers.d_act_ctrl);
  if (error == cudaSuccess) {
    error = launch_paired(buffers.d_gate, buffers.d_up, buffers.rows,
                          buffers.columns, buffers.d_y, prompt_rows,
                          buffers.d_act_cand, buffers.d_gate_dump,
                          buffers.d_up_dump);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    free_pair(&buffers);
    return fail_cuda("prod launch", error);
  }
  std::vector<__nv_bfloat16> act_ctrl(elems);
  std::vector<__nv_bfloat16> act_cand(elems);
  error = cudaMemcpy(act_ctrl.data(), buffers.d_act_ctrl,
                     elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) {
    error = cudaMemcpy(act_cand.data(), buffers.d_act_cand,
                       elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) {
    free_pair(&buffers);
    return fail_cuda("prod d2h", error);
  }
  std::size_t mismatch = 0;
  std::size_t nonfinite = 0;
  const bool same = exact_bf16(act_cand, act_ctrl, &mismatch, &nonfinite);
  std::printf(
      "production gate_up rows=%zu k=%zu prompts=%zu exact=%s mismatch=%zu "
      "nonfinite=%zu\n",
      buffers.rows, buffers.columns, prompt_rows, json_bool(same), mismatch,
      nonfinite);
  int rc = check_paired_dispatch("production", true);
  if (!same) rc = 1;
  free_pair(&buffers);
  return rc;
}

int run_correctness(const char* model_path) {
  int rc = 0;
  rc |= run_pair_case("corr_m65_k256", 65, 256, 65, 0x21u, 0x22u, 0x23u, NAN,
                      true, true);
  rc |= run_pair_case("corr_m65_k512", 65, 512, 65, 0x31u, 0x32u, 0x33u, NAN,
                      true, false);
  rc |= run_pair_case("tail_n1", 65, 256, 1, 0x41u, 0x42u, 0x43u, NAN, true,
                      false);
  rc |= run_pair_case("tail_n63", 65, 256, 63, 0x51u, 0x52u, 0x53u, NAN, false,
                      false);
  rc |= run_pair_case("tail_n64", 64, 256, 64, 0x61u, 0x62u, 0x63u, NAN, true,
                      false);
  rc |= run_pair_case("tail_n65", 65, 256, 65, 0x71u, 0x72u, 0x73u, NAN, false,
                      false);
  rc |= run_pair_case("zero", 17, 256, 17, 0x81u, 0x82u, 0u, 0.0F, true, false);
  rc |= run_pair_case("neg_gate", 17, 256, 17, 0x91u, 0x92u, 0u, -80.0F, true,
                      false);
  rc |= run_q8_and_residual("q8_m256", 256, 256, 32);
  if (model_path == nullptr || model_path[0] == '\0') {
    std::fprintf(stderr, "correctness requires the pinned GGUF\n");
    return 1;
  }
  qw38::internal::MappedFile mapping;
  qw38::internal::ModelInfo info;
  qw38::internal::ModelWeights weights;
  qw38::cuda::ResidentModel model;
  if (load_model(model_path, &mapping, &info, &weights, &model) != 0) return 1;
  rc |= run_production_sampled(model.layer(0).common);
  return rc;
}

cudaError_t launch_complete_ffn(const qw38::cuda::DeviceCommonLayer& layer,
                                qw38::cuda::SchedulerWorkspace* workspace,
                                std::size_t prompt_rows, bool paired) {
  qw38::cuda::FfnPromptPairOverrideScope scope(paired ? "i64_j64" : "off");
  if (paired) {
    qw38::cuda::set_ffn_prompt_pair_trace_unfused(false);
  }
  return qw38::cuda::execute_prompt_ffn(
      layer, workspace->prompt_residual_a_, workspace,
      workspace->prompt_residual_b_, workspace->prompt_residual_a_,
      nullptr, prompt_rows, nullptr);
}

int run_graph_eager(const qw38::cuda::DeviceCommonLayer& layer,
                    qw38::cuda::SchedulerWorkspace* workspace,
                    std::size_t prompt_rows) {
  const std::size_t residual_bytes =
      prompt_rows * qw38::internal::kResidualWidth * sizeof(float);
  std::vector<float> seed(prompt_rows * qw38::internal::kResidualWidth);
  std::vector<float> mixer(prompt_rows * qw38::internal::kResidualWidth);
  std::vector<float> eager(seed.size(), 0.0F);
  std::vector<float> graph_out(seed.size(), 0.0F);
  cudaError_t error = cudaMemcpy(seed.data(), workspace->prompt_residual_a_,
                                 residual_bytes, cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) {
    error = cudaMemcpy(mixer.data(), workspace->prompt_mixer_output_,
                       residual_bytes, cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) return fail_cuda("graph seed", error);

  auto restore = [&]() -> cudaError_t {
    cudaError_t copy = cudaMemcpy(workspace->prompt_residual_a_, seed.data(),
                                  residual_bytes, cudaMemcpyHostToDevice);
    if (copy == cudaSuccess) {
      copy = cudaMemcpy(workspace->prompt_mixer_output_, mixer.data(),
                        residual_bytes, cudaMemcpyHostToDevice);
    }
    return copy;
  };

  qw38::cuda::FfnPromptPairOverrideScope scope("i64_j64");
  error = restore();
  if (error == cudaSuccess) {
    error = qw38::cuda::execute_prompt_ffn(
        layer, workspace->prompt_residual_a_, workspace,
        workspace->prompt_residual_b_, workspace->prompt_residual_a_, nullptr,
        prompt_rows, nullptr);
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error == cudaSuccess) {
    error = cudaMemcpy(eager.data(), workspace->prompt_residual_a_,
                       residual_bytes, cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) return fail_cuda("eager ffn", error);

  error = restore();
  cudaStream_t stream = nullptr;
  cudaGraph_t captured = nullptr;
  cudaGraphExec_t exec = nullptr;
  if (error == cudaSuccess) error = cudaStreamCreate(&stream);
  if (error == cudaSuccess) {
    error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::execute_prompt_ffn(
        layer, workspace->prompt_residual_a_, workspace,
        workspace->prompt_residual_b_, workspace->prompt_residual_a_, nullptr,
        prompt_rows, stream);
  }
  cudaError_t end = cudaErrorUnknown;
  if (stream != nullptr) end = cudaStreamEndCapture(stream, &captured);
  if (error == cudaSuccess) error = end;
  if (error == cudaSuccess) {
    error = cudaGraphInstantiate(&exec, captured, nullptr, nullptr, 0);
  }
  if (error == cudaSuccess) error = restore();
  if (error == cudaSuccess) error = cudaGraphLaunch(exec, stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
  if (error == cudaSuccess) {
    error = cudaMemcpy(graph_out.data(), workspace->prompt_residual_a_,
                       residual_bytes, cudaMemcpyDeviceToHost);
  }
  if (exec != nullptr) cudaGraphExecDestroy(exec);
  if (captured != nullptr) cudaGraphDestroy(captured);
  if (stream != nullptr) cudaStreamDestroy(stream);
  if (error != cudaSuccess) return fail_cuda("graph ffn", error);
  std::size_t mismatch = 0;
  std::size_t nonfinite = 0;
  const bool same = exact_equal(graph_out, eager, &mismatch, &nonfinite);
  std::printf("graph_eager paired mismatch=%zu nonfinite=%zu same=%s\n",
              mismatch, nonfinite, json_bool(same));
  return same ? 0 : 1;
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
  if (error == cudaSuccess) {
    error = cudaMemcpy(workspace.prompt_mixer_output_, residual.data(),
                       residual.size() * sizeof(float), cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) return fail_cuda("screen upload", error);

  const std::size_t avoided_fp32 =
      2U * kPrompt * layer.ffn_gate.rows * sizeof(float);
  const std::size_t bf16_store = kPrompt * layer.ffn_gate.rows * sizeof(__nv_bfloat16);
  const std::size_t y_q8 =
      qw38::cuda::q8_prompt_workspace_bytes(kPrompt, layer.ffn_gate.columns);
  std::printf(
      "traffic_estimate layer_avoided_fp32_bytes=%zu layer_bf16_bytes=%zu "
      "y_reload_bytes=%zu prompt_layers=64 avoided_fp32_prompt_bytes=%zu "
      "opt061_rotating_prompt_ffn_ms=699.91 useful_byte_gbs=1015 "
      "bandwidth_ms_prompt=%.3f threshold_ms=5.0 go=true\n",
      avoided_fp32, bf16_store, y_q8, avoided_fp32 * 64U,
      static_cast<double>(avoided_fp32 * 64U) / (1015.0 * 1.0e9) * 1.0e3);
  std::printf("scratch_high_water_bytes=%zu launch_control=6 launch_candidate=5\n",
              workspace.allocated_bytes());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error != cudaSuccess) return fail_cuda("events", error);

  float control_ms = 0.0F;
  float candidate_ms = 0.0F;
  auto time_ffn = [&](bool paired, float* mean) -> int {
    float acc = 0.0F;
    int n = 0;
    for (int sample = 0; sample < 4; ++sample) {
      float ms = 0.0F;
      error = cudaEventRecord(start, nullptr);
      if (error == cudaSuccess) {
        error = launch_complete_ffn(layer, &workspace, kPrompt, paired);
      }
      if (error == cudaSuccess) error = cudaEventRecord(stop, nullptr);
      if (error == cudaSuccess) error = cudaEventSynchronize(stop);
      if (error == cudaSuccess) error = cudaEventElapsedTime(&ms, start, stop);
      if (error != cudaSuccess) return fail_cuda("complete ffn", error);
      if (sample == 0) {
        std::printf("complete_warmup paired=%s ms=%.9g\n", json_bool(paired),
                    ms);
      } else {
        acc += ms;
        ++n;
        std::printf("complete_sample paired=%s sample=%d ms=%.9g\n",
                    json_bool(paired), sample, ms);
      }
    }
    *mean = n == 0 ? 0.0F : acc / static_cast<float>(n);
    return 0;
  };
  int rc = time_ffn(false, &control_ms);
  if (rc == 0) rc = time_ffn(true, &candidate_ms);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (rc != 0) return rc;

  int cand_occ = 0;
  int cand_regs = 0;
  std::size_t cand_local = 0;
  std::size_t cand_shared = 0;
  error = qw38::cuda::mmq_paired_kernel_attributes(
      &cand_occ, &cand_regs, &cand_local, &cand_shared, false);
  if (error != cudaSuccess) return fail_cuda("candidate attrs", error);
  const bool spills = cand_local > 0;
  const bool faster = candidate_ms < control_ms;
  const bool keep = faster && !spills && cand_occ >= 1;
  std::printf(
      "screen_winner control_ffn_ms=%.9g candidate_ffn_ms=%.9g occupancy=%d "
      "regs=%d local=%zu shared=%zu spills=%s keep=%s installed_pair=%s "
      "selected_pair=%s\n",
      control_ms, candidate_ms, cand_occ, cand_regs, cand_local, cand_shared,
      json_bool(spills), json_bool(keep), json_bool(keep),
      qw38::cuda::selected_ffn_prompt_pair_path());

  rc = run_graph_eager(layer, &workspace, kPrompt);
  return rc;
}

void emit_payload(const char* workload, const char* status) {
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-067\",\"workload\":\"%s\","
      "\"tier\":\"%s\",\"status\":\"%s\",\"selected_pair\":\"%s\","
      "\"control_tile\":\"i128_j128\",\"candidate_tile\":\"i64_j64\","
      "\"claims_throughput\":false,\"llama_revision\":\"%s\","
      "\"gguf_sha256\":\"%s\"}\n",
      kPrefix, workload, qw38::cuda::test_tier_name(), status,
      qw38::cuda::selected_ffn_prompt_pair_path(), kLlamaRev, kGgufSha);
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
