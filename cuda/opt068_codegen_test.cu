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

constexpr char kPrefix[] = "QW38_OPT068_CODEGEN_RESULT=";
constexpr const char* kFamily = "q4_prompt_mmq";
constexpr const char* kGgufSha =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr const char* kLlamaRev =
    "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr std::size_t kQ4KBytes = 144;
constexpr std::size_t kQ4KValues = 256;
constexpr float kAbsScale = 0.20F;
constexpr float kRelTol = 0.05F;
constexpr int kHeldOutRows = 16;
constexpr int kHeldOutPrompts = 4;
constexpr unsigned int kQualityI = 128;
constexpr unsigned int kPromptTile = 128;
constexpr std::size_t kHeldOutLayers[2] = {62, 63};

struct Options final {
  const char* workload = nullptr;
  const char* model = nullptr;
};

const char* json_bool(bool value) { return value ? "true" : "false"; }

const char* variant_from_path(const char* argv0) {
  if (argv0 != nullptr && std::strstr(argv0, "o3_fmad_true") != nullptr) {
    return "o3_fmad_true";
  }
  if (argv0 != nullptr && std::strstr(argv0, "o3_fmad_false") != nullptr) {
    return "o3_fmad_false";
  }
  return "o2_fmad_false";
}

std::string sibling(const char* argv0, const char* name) {
  std::string path = argv0 != nullptr ? argv0 : "";
  const auto slash = path.find_last_of('/');
  if (slash == std::string::npos) return name;
  return path.substr(0, slash + 1) + name;
}

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

void fill_prompt(std::size_t prompt_rows, std::size_t columns,
                 std::vector<__nv_bfloat16>* prompt, std::uint32_t seed,
                 bool cancel) {
  prompt->resize(prompt_rows * columns);
  for (std::size_t index = 0; index < prompt->size(); ++index) {
    float value = unit_normal(static_cast<std::uint32_t>(index), seed);
    if (cancel) value = (index % 2U == 0) ? value : -value;
    (*prompt)[index] = __float2bfloat16_rn(value);
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

int launch_q4(const std::uint8_t* weights, std::size_t output_rows,
              std::size_t columns, const qw38::cuda::Q8Block* y,
              std::size_t prompt_rows, float* output) {
  qw38::cuda::clear_mmq_tile_dispatch();
  const cudaError_t error = qw38::cuda::launch_quant_mmq_mma_y_ij(
      qw38::cuda::QuantKind::kQ4K, weights, output_rows, columns, y,
      prompt_rows, output, kQualityI, kPromptTile, nullptr);
  if (error != cudaSuccess) return fail_cuda("q4 launch", error);
  return 0;
}

int inspect_codegen(const char* variant, const char* object_path,
                    const char* stamp_path) {
  int occupancy = 0;
  int registers = 0;
  std::size_t local_bytes = 0;
  std::size_t shared_bytes = 0;
  cudaError_t error = qw38::cuda::mmq_pipeline_kernel_attributes(
      qw38::cuda::QuantKind::kQ4K, kPromptTile, kQualityI, "fma_async",
      &occupancy, &registers, &local_bytes, &shared_bytes);
  int x_occ = 0;
  int x_regs = 0;
  std::size_t x_local = 0;
  std::size_t x_shared = 0;
  const cudaError_t x_error = qw38::cuda::mmq_x_pipeline_kernel_attributes(
      &x_occ, &x_regs, &x_local, &x_shared);
  std::printf(
      "codegen family=%s variant=%s occupancy=%d regs=%d local=%zu "
      "shared=%zu x_occupancy=%d x_regs=%d x_local=%zu x_shared=%zu "
      "object=%s stamp=%s query=%s x_query=%s "
      "explicit_fma_keeps=noncontracting\n",
      kFamily, variant, occupancy, registers, local_bytes, shared_bytes, x_occ,
      x_regs, x_local, x_shared, object_path, stamp_path,
      error == cudaSuccess ? "ok" : cudaGetErrorString(error),
      x_error == cudaSuccess ? "ok" : cudaGetErrorString(x_error));
  if (error != cudaSuccess || occupancy < 1) return 1;

  FILE* dump = popen(
      (std::string("cuobjdump -sass ") + object_path +
       " 2>/dev/null | awk '"
       "BEGIN{ffma=0;fmul=0;fadd=0} "
       "/FFMA/{ffma++} /FMUL/{fmul++} /FADD/{fadd++} "
       "END{printf \"sass_counts ffma=%d fmul=%d fadd=%d\\n\",ffma,fmul,fadd}'")
          .c_str(),
      "r");
  if (dump != nullptr) {
    char line[256];
    if (std::fgets(line, sizeof(line), dump) != nullptr) {
      std::fputs(line, stdout);
    } else {
      std::printf("sass_counts ffma=-1 fmul=-1 fadd=-1 reason=unavailable\n");
    }
    pclose(dump);
  } else {
    std::printf("sass_counts ffma=-1 fmul=-1 fadd=-1 reason=popen_failed\n");
  }
  return 0;
}

int run_q4_case(const char* name, std::size_t rows, std::size_t columns,
                std::size_t prompt_rows, bool cancel, bool require_aligned) {
  std::vector<std::uint8_t> weights;
  fill_q4_weights(rows, columns, &weights);
  std::vector<__nv_bfloat16> prompt;
  fill_prompt(prompt_rows, columns, &prompt, 0x68u, cancel);
  std::vector<float> expected;
  if (!reference_dequant_gemm(weights, rows, columns, prompt, prompt_rows,
                              all_indices(rows), all_indices(prompt_rows),
                              &expected)) {
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
        qw38::cuda::QuantKind::kQ4K, d_p, prompt_rows, columns, d_y, nullptr);
  }
  if (error != cudaSuccess) {
    cudaFree(d_o);
    cudaFree(d_y);
    cudaFree(d_p);
    cudaFree(d_w);
    return fail_cuda("case alloc", error);
  }
  int rc = launch_q4(d_w, rows, columns, d_y, prompt_rows, d_o);
  if (rc == 0) {
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) rc = fail_cuda("case sync", error);
  }
  std::vector<float> actual(prompt_rows * rows);
  if (rc == 0) {
    error = cudaMemcpy(actual.data(), d_o, actual.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) rc = fail_cuda("case d2h", error);
  }
  float max_abs = 0.0F;
  std::size_t bad = 0;
  std::size_t nonfinite = 0;
  const bool ok =
      rc == 0 && ds4_ok(actual, expected, columns, &max_abs, &bad, &nonfinite);
  const qw38::cuda::MmqTileDispatch rec = qw38::cuda::last_mmq_tile_dispatch();
  std::printf(
      "case name=%s rows=%zu cols=%zu prompt=%zu cancel=%s aligned=%s "
      "pipeline=%s kernel=%s max_abs=%.9g bad=%zu nonfinite=%zu\n",
      name, rows, columns, prompt_rows, json_bool(cancel),
      json_bool(rec.aligned), json_bool(rec.pipeline), rec.kernel, max_abs, bad,
      nonfinite);
  if (require_aligned && (!rec.aligned || !rec.pipeline)) rc = 1;
  if (!ok) rc = 1;
  cudaFree(d_o);
  cudaFree(d_y);
  cudaFree(d_p);
  cudaFree(d_w);
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

int run_held_out(const qw38::cuda::DeviceTensor& tensor, const char* name) {
  const std::vector<std::size_t> rows =
      sample_indices(tensor.rows, kHeldOutRows);
  std::vector<std::uint8_t> packed;
  if (copy_q4_rows(tensor, rows, &packed) != 0) return 1;
  std::vector<__nv_bfloat16> prompt;
  fill_prompt(kHeldOutPrompts, tensor.columns, &prompt, 0x590u, false);
  std::vector<float> expected;
  if (!reference_dequant_gemm(packed, rows.size(), tensor.columns, prompt,
                              kHeldOutPrompts, all_indices(rows.size()),
                              all_indices(kHeldOutPrompts), &expected)) {
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
                                 kHeldOutPrompts, tensor.columns));
  }
  if (error == cudaSuccess) {
    error =
        cudaMalloc(&d_o, kHeldOutPrompts * rows.size() * sizeof(float));
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
        qw38::cuda::QuantKind::kQ4K, d_p, kHeldOutPrompts, tensor.columns, d_y,
        nullptr);
  }
  if (error != cudaSuccess) {
    cudaFree(d_o);
    cudaFree(d_y);
    cudaFree(d_p);
    cudaFree(d_w);
    return fail_cuda("held-out alloc", error);
  }
  int rc = launch_q4(d_w, rows.size(), tensor.columns, d_y, kHeldOutPrompts,
                     d_o);
  if (rc == 0) {
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) rc = fail_cuda("held-out sync", error);
  }
  std::vector<float> actual(kHeldOutPrompts * rows.size());
  if (rc == 0) {
    error = cudaMemcpy(actual.data(), d_o, actual.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) rc = fail_cuda("held-out d2h", error);
  }
  float max_abs = 0.0F;
  std::size_t bad = 0;
  std::size_t nonfinite = 0;
  const bool ok =
      rc == 0 &&
      ds4_ok(actual, expected, tensor.columns, &max_abs, &bad, &nonfinite);
  std::printf(
      "held_out name=%s rows=%zu k=%zu prompts=%d max_abs=%.9g bad=%zu "
      "nonfinite=%zu shared_reference=fp64_decode\n",
      name, rows.size(), tensor.columns, kHeldOutPrompts, max_abs, bad,
      nonfinite);
  if (!ok) rc = 1;
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

int run_smoke(const char* variant, const char* object_path,
              const char* stamp_path) {
  int rc = inspect_codegen(variant, object_path, stamp_path);
  rc |= run_q4_case("smoke_m17", 17, 256, 17, false, false);
  return rc;
}

int run_correctness(const char* model_path) {
  int rc = run_q4_case("corr_random", 128, 256, 128, false, true);
  rc |= run_q4_case("corr_cancel", 128, 256, 128, true, true);
  if (model_path == nullptr || model_path[0] == '\0') {
    std::fprintf(stderr, "correctness requires the pinned GGUF\n");
    return 1;
  }
  qw38::internal::MappedFile mapping;
  qw38::internal::ModelInfo info;
  qw38::internal::ModelWeights weights;
  qw38::cuda::ResidentModel model;
  if (load_model(model_path, &mapping, &info, &weights, &model) != 0) return 1;
  for (std::size_t layer : kHeldOutLayers) {
    const qw38::cuda::DeviceCommonLayer& common = model.layer(layer).common;
    char gate_name[32];
    char down_name[32];
    std::snprintf(gate_name, sizeof(gate_name), "L%zu_gate_k5120", layer);
    std::snprintf(down_name, sizeof(down_name), "L%zu_down_k17408", layer);
    rc |= run_held_out(common.ffn_gate, gate_name);
    rc |= run_held_out(common.ffn_down, down_name);
  }
  return rc;
}

cudaError_t launch_complete_ffn(const qw38::cuda::DeviceCommonLayer& layer,
                                qw38::cuda::SchedulerWorkspace* workspace,
                                std::size_t prompt_rows) {
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

int run_screen(const char* model_path, const char* variant) {
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
  fill_prompt(kPrompt, qw38::internal::kResidualWidth, &prompt, 0x4096u, false);
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

  float acc = 0.0F;
  int n = 0;
  for (int sample = 0; sample < 4; ++sample) {
    float ms = 0.0F;
    error = cudaEventRecord(start, nullptr);
    if (error == cudaSuccess) {
      error = launch_complete_ffn(layer, &workspace, kPrompt);
    }
    if (error == cudaSuccess) error = cudaEventRecord(stop, nullptr);
    if (error == cudaSuccess) error = cudaEventSynchronize(stop);
    if (error == cudaSuccess) error = cudaEventElapsedTime(&ms, start, stop);
    if (error != cudaSuccess) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      return fail_cuda("complete ffn", error);
    }
    const qw38::cuda::MmqTileDispatch rec = qw38::cuda::last_mmq_tile_dispatch();
    if (sample == 0) {
      std::printf("complete_warmup variant=%s ms=%.9g kernel=%s pipeline=%s\n",
                  variant, ms, rec.kernel, json_bool(rec.pipeline));
    } else {
      acc += ms;
      ++n;
      std::printf("complete_sample variant=%s sample=%d ms=%.9g kernel=%s\n",
                  variant, sample, ms, rec.kernel);
    }
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  const float mean = n == 0 ? 0.0F : acc / static_cast<float>(n);
  std::printf("complete_ffn_ms variant=%s ms=%.9g layers=1 prompt=4096 "
              "family=%s rotating=layer0\n",
              variant, mean, kFamily);
  std::printf(
      "QW38_OPT068_VARIANT_RESULT={\"schema_version\":1,\"task\":\"OPT-068\","
      "\"variant\":\"%s\",\"family\":\"%s\",\"complete_ffn_ms\":%.9g,"
      "\"tile\":\"i128_j128\",\"claims_throughput\":false}\n",
      variant, kFamily, mean);
  return 0;
}

void emit_payload(const char* workload, const char* variant,
                  const char* status) {
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-068\",\"workload\":\"%s\","
      "\"tier\":\"%s\",\"status\":\"%s\",\"family\":\"%s\",\"variant\":\"%s\","
      "\"tile\":\"i128_j128\",\"claims_throughput\":false,"
      "\"llama_revision\":\"%s\",\"gguf_sha256\":\"%s\"}\n",
      kPrefix, workload, qw38::cuda::test_tier_name(), status, kFamily, variant,
      kLlamaRev, kGgufSha);
}

}  // namespace

int main(int argc, char** argv) {
  if (!qw38::cuda::test_tier_valid()) {
    std::fprintf(stderr, "QW38_CUDA_TEST_TIER must be set\n");
    return 2;
  }
  Options options;
  if (parse_args(argc, argv, &options) != 0) return 2;
  const char* variant = variant_from_path(argv[0]);
  const std::string object = sibling(argv[0], "q4_prompt_mmq.cuda.o");
  const std::string stamp = sibling(argv[0], "nvccflags.stamp");
  const char* workload = options.workload != nullptr
                             ? options.workload
                             : default_workload(qw38::cuda::test_tier());
  std::printf("opt068_family=%s variant=%s object=%s stamp=%s "
              "opt061_rank=prompt-ffn\n",
              kFamily, variant, object.c_str(), stamp.c_str());
  int rc = 0;
  if (std::strcmp(workload, "smoke") == 0) {
    rc = run_smoke(variant, object.c_str(), stamp.c_str());
  } else if (std::strcmp(workload, "correctness") == 0) {
    rc = run_correctness(options.model);
  } else if (std::strcmp(workload, "screen") == 0 ||
             std::strcmp(workload, "acceptance") == 0) {
    rc = run_screen(options.model, variant);
  } else {
    return usage(argv[0]);
  }
  emit_payload(workload, variant, rc == 0 ? "passed" : "failed");
  std::printf("status=%s variant=%s\n", rc == 0 ? "passed" : "failed", variant);
  return rc;
}
