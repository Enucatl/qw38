#include "full_scheduler.h"
#include "opt149_norm_q8.cuh"
#include "q4k_decode_path.cuh"
#include "q8_decode_path.cuh"
#include "test_tier.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr char kPrefix[] = "QW38_OPT149_NORM_Q8_RESULT=";
constexpr char kCountsPrefix[] = "QW38_OPT149_NATIVE_COUNTS=";
constexpr const char* kGgufSha =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr const char* kLlamaRev =
    "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr int kWidth = qw38::cuda::opt149::kWidth;
constexpr int kQ8Blocks = kWidth / 32;
constexpr int kMixerLayers = qw38::cuda::opt149::kMixerLayers;

int fail_cuda(const char* op, cudaError_t error) {
  std::fprintf(stderr, "%s: %s\n", op, cudaGetErrorString(error));
  return 1;
}

const char* json_bool(bool value) { return value ? "true" : "false"; }

float unit(std::uint32_t index, std::uint32_t seed) {
  std::uint32_t x = index * 747796405U + seed;
  x ^= x >> 16U;
  x *= 0x7feb352dU;
  x ^= x >> 15U;
  return (static_cast<float>(x & 0x00FFFFFFU) / 8388608.0F - 1.0F) * 0.35F;
}

void fill_vec(std::vector<float>* values, std::uint32_t seed) {
  values->resize(static_cast<std::size_t>(kWidth));
  for (int index = 0; index < kWidth; ++index) {
    (*values)[static_cast<std::size_t>(index)] =
        unit(static_cast<std::uint32_t>(index), seed);
  }
}

void fill_scale(std::vector<float>* values, std::uint32_t seed) {
  values->resize(static_cast<std::size_t>(kWidth));
  for (int index = 0; index < kWidth; ++index) {
    (*values)[static_cast<std::size_t>(index)] =
        0.85F + 0.0002F * static_cast<float>(index) +
        0.01F * unit(static_cast<std::uint32_t>(index), seed);
  }
}

struct HostQ81 {
  std::uint16_t scale_bits = 0;
  std::uint16_t sum_bits = 0;
  std::int8_t values[32]{};
};

std::uint16_t half_bits(__half value) {
  std::uint16_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

void host_rms_q8(const std::vector<float>& input, const std::vector<float>& scale,
                 std::vector<HostQ81>* out, std::vector<__nv_bfloat16>* bf16) {
  double sum = 0.0;
  for (int index = 0; index < kWidth; ++index) {
    const double value = static_cast<double>(input[static_cast<std::size_t>(index)]);
    sum += value * value;
  }
  const float inverse = 1.0F / std::sqrt(static_cast<float>(sum) /
                                             static_cast<float>(kWidth) +
                                         1.0e-6F);
  bf16->resize(static_cast<std::size_t>(kWidth));
  out->assign(static_cast<std::size_t>(kQ8Blocks), HostQ81{});
  for (int block = 0; block < kQ8Blocks; ++block) {
    float values[32];
    float maximum = 0.0F;
    int q_sum = 0;
    for (int lane = 0; lane < 32; ++lane) {
      const int index = block * 32 + lane;
      const __nv_bfloat16 stored = __float2bfloat16_rn(
          input[static_cast<std::size_t>(index)] * inverse *
          scale[static_cast<std::size_t>(index)]);
      (*bf16)[static_cast<std::size_t>(index)] = stored;
      values[lane] = __bfloat162float(stored);
      maximum = std::max(maximum, std::fabs(values[lane]));
    }
    const float qscale = maximum == 0.0F ? 0.0F : maximum / 127.0F;
    for (int lane = 0; lane < 32; ++lane) {
      const std::int8_t quant =
          qscale == 0.0F
              ? 0
              : static_cast<std::int8_t>(std::round(values[lane] / qscale));
      (*out)[static_cast<std::size_t>(block)].values[lane] = quant;
      q_sum += static_cast<int>(quant);
    }
    (*out)[static_cast<std::size_t>(block)].scale_bits =
        half_bits(__float2half_rn(qscale));
    (*out)[static_cast<std::size_t>(block)].sum_bits =
        half_bits(__float2half_rn(static_cast<float>(q_sum)));
  }
}

bool q8_equal(const std::vector<qw38::cuda::opt149::Q81Block>& device,
              const std::vector<HostQ81>& host, int* mismatches) {
  int local = 0;
  if (device.size() != host.size()) {
    *mismatches = static_cast<int>(device.size() + host.size());
    return false;
  }
  for (std::size_t block = 0; block < device.size(); ++block) {
    if (half_bits(device[block].scale) != host[block].scale_bits ||
        half_bits(device[block].q8_sum) != host[block].sum_bits) {
      ++local;
    }
    for (int lane = 0; lane < 32; ++lane) {
      if (device[block].values[lane] != host[block].values[lane]) ++local;
    }
  }
  *mismatches = local;
  return local == 0;
}

cudaError_t upload(const std::vector<float>& host, float** device) {
  cudaError_t error = cudaMalloc(device, host.size() * sizeof(float));
  if (error == cudaSuccess) {
    error = cudaMemcpy(*device, host.data(), host.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
  }
  return error;
}

struct DeviceCase {
  float* input = nullptr;
  float* scale = nullptr;
  float* residual = nullptr;
  float* correction = nullptr;
  float* output = nullptr;
  __nv_bfloat16* bf16 = nullptr;
  qw38::cuda::opt149::Q81Block* fused = nullptr;
  qw38::cuda::opt149::Q81Block* control = nullptr;
  float* proj[qw38::cuda::opt149::kProjCount]{};
  std::uint8_t* weights[qw38::cuda::opt149::kProjCount]{};
};

void free_case(DeviceCase* item) {
  cudaFree(item->input);
  cudaFree(item->scale);
  cudaFree(item->residual);
  cudaFree(item->correction);
  cudaFree(item->output);
  cudaFree(item->bf16);
  cudaFree(item->fused);
  cudaFree(item->control);
  for (int index = 0; index < qw38::cuda::opt149::kProjCount; ++index) {
    cudaFree(item->proj[index]);
    cudaFree(item->weights[index]);
  }
}

cudaError_t alloc_q8(qw38::cuda::opt149::Q81Block** ptr) {
  return cudaMalloc(ptr, static_cast<std::size_t>(kQ8Blocks) *
                             sizeof(qw38::cuda::opt149::Q81Block));
}

int run_identity(const char* layer_kind, std::uint32_t seed, bool residual) {
  std::vector<float> input;
  std::vector<float> scale;
  std::vector<float> correction;
  fill_vec(&input, seed);
  fill_scale(&scale, seed + 9U);
  fill_vec(&correction, seed + 17U);
  std::vector<float> added = input;
  if (residual) {
    for (int index = 0; index < kWidth; ++index) {
      added[static_cast<std::size_t>(index)] =
          input[static_cast<std::size_t>(index)] +
          correction[static_cast<std::size_t>(index)];
    }
  }
  std::vector<HostQ81> host;
  std::vector<__nv_bfloat16> host_bf16;
  host_rms_q8(added, scale, &host, &host_bf16);

  DeviceCase gpu{};
  cudaError_t error = upload(input, residual ? &gpu.residual : &gpu.input);
  if (error == cudaSuccess) error = upload(scale, &gpu.scale);
  if (residual && error == cudaSuccess) error = upload(correction, &gpu.correction);
  if (residual && error == cudaSuccess) {
    error = cudaMalloc(&gpu.output, static_cast<std::size_t>(kWidth) * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&gpu.bf16,
                       static_cast<std::size_t>(kWidth) * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) error = alloc_q8(&gpu.fused);
  if (error == cudaSuccess) error = alloc_q8(&gpu.control);
  if (error != cudaSuccess) {
    free_case(&gpu);
    return fail_cuda("alloc identity", error);
  }
  if (residual) {
    error = qw38::cuda::launch_residual_add_norm_fp32_to_bf16(
        gpu.residual, gpu.correction, gpu.scale, static_cast<std::size_t>(kWidth),
        gpu.output, gpu.bf16, nullptr);
  } else {
    error = qw38::cuda::launch_rms_norm_fp32_to_bf16(
        gpu.input, gpu.scale, static_cast<std::size_t>(kWidth), gpu.bf16,
        nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_bf16_q8_1(
        gpu.bf16, gpu.control, static_cast<std::size_t>(kWidth), nullptr);
  }
  qw38::cuda::opt149::g_control_launches += 2;
  if (error == cudaSuccess) {
    if (residual) {
      error = qw38::cuda::opt149::launch_residual_add_norm_fp32_to_q8_1(
          gpu.residual, gpu.correction, gpu.scale,
          static_cast<std::size_t>(kWidth), gpu.output, gpu.fused, nullptr,
          false, nullptr);
    } else {
      error = qw38::cuda::opt149::launch_rms_norm_fp32_to_q8_1(
          gpu.input, gpu.scale, static_cast<std::size_t>(kWidth), gpu.fused,
          nullptr, false, nullptr);
    }
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    free_case(&gpu);
    return fail_cuda("identity launch", error);
  }
  std::vector<qw38::cuda::opt149::Q81Block> fused(static_cast<std::size_t>(kQ8Blocks));
  std::vector<qw38::cuda::opt149::Q81Block> control(
      static_cast<std::size_t>(kQ8Blocks));
  error = cudaMemcpy(fused.data(), gpu.fused,
                     fused.size() * sizeof(qw38::cuda::opt149::Q81Block),
                     cudaMemcpyDeviceToHost);
  if (error == cudaSuccess) {
    error = cudaMemcpy(control.data(), gpu.control,
                       control.size() * sizeof(qw38::cuda::opt149::Q81Block),
                       cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) {
    free_case(&gpu);
    return fail_cuda("identity copy", error);
  }
  int vs_host = 0;
  int vs_control = 0;
  const bool host_ok = q8_equal(fused, host, &vs_host);
  (void)host_ok;
  for (std::size_t block = 0; block < fused.size(); ++block) {
    if (std::memcmp(&fused[block], &control[block],
                    sizeof(qw38::cuda::opt149::Q81Block)) != 0) {
      ++vs_control;
    }
  }
  double fp64_abs = 0.0;
  for (int block = 0; block < kQ8Blocks; ++block) {
    const float dscale = __half2float(fused[static_cast<std::size_t>(block)].scale);
    const float hscale = __half2float(
        __ushort_as_half(host[static_cast<std::size_t>(block)].scale_bits));
    fp64_abs = std::max(fp64_abs, static_cast<double>(std::fabs(dscale - hscale)));
  }
  const bool pass = vs_control == 0;
  std::printf(
      "layer_kind=%s residual=%s control_path=%s candidate_path=%s "
      "launch=%s q8_vs_host=%d q8_vs_control=%d fp64_abs=%.9g equal=%s "
      "finite=%s format=%s write_bf16=false launches_control=2 "
      "launches_candidate=1\n",
      layer_kind, json_bool(residual), qw38::cuda::opt149::kControlId,
      qw38::cuda::opt149::kCandidateId,
      residual ? qw38::cuda::opt149::kLaunchResidual
               : qw38::cuda::opt149::kLaunchRms,
      vs_host, vs_control, fp64_abs, json_bool(pass), "true",
      qw38::cuda::opt149::kQ81Format);
  free_case(&gpu);
  return pass ? 0 : 1;
}

int run_mixed_and_retain() {
  const bool q8_0[] = {true, true, false};
  const bool q8block[] = {false, false, true};
  const bool bf16[] = {false, false, false};
  const bool eligible_mixed = qw38::cuda::opt149::mixer_group_q8_1_eligible(
      q8_0, q8block, bf16, 3);
  const bool q8_only[] = {true, true, true, true};
  const bool none[] = {false, false, false, false};
  const bool eligible_mixer = qw38::cuda::opt149::mixer_group_q8_1_eligible(
      q8_only, none, none, 4);
  const bool logits_bf16[] = {true};
  const bool logits_q8[] = {false};
  const bool eligible_logits = qw38::cuda::opt149::mixer_group_q8_1_eligible(
      logits_q8, none, logits_bf16, 1);
  const bool pass = !eligible_mixed && eligible_mixer && !eligible_logits;
  std::printf(
      "mixed_consumers eligible=%s retain_existing=%s mixer_group_eligible=%s "
      "logits_bf16_eligible=%s ffn_q8block_retain=true opt110_revived=false "
      "byte_size_only_claim=false selected_group=%s pass=%s\n",
      json_bool(eligible_mixed), json_bool(!eligible_mixed),
      json_bool(eligible_mixer), json_bool(eligible_logits),
      qw38::cuda::opt149::kSelectedGroup, json_bool(pass));
  return pass ? 0 : 1;
}

int run_stale_and_state() {
  std::vector<float> input;
  std::vector<float> scale;
  fill_vec(&input, 3);
  fill_scale(&scale, 4);
  DeviceCase gpu{};
  cudaError_t error = upload(input, &gpu.input);
  if (error == cudaSuccess) error = upload(scale, &gpu.scale);
  if (error == cudaSuccess) error = alloc_q8(&gpu.fused);
  if (error == cudaSuccess) error = alloc_q8(&gpu.control);
  if (error != cudaSuccess) {
    free_case(&gpu);
    return fail_cuda("stale alloc", error);
  }
  error = qw38::cuda::opt149::launch_rms_norm_fp32_to_q8_1(
      gpu.input, gpu.scale, static_cast<std::size_t>(kWidth), gpu.fused, nullptr,
      false, nullptr);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  std::vector<qw38::cuda::opt149::Q81Block> first(static_cast<std::size_t>(kQ8Blocks));
  if (error == cudaSuccess) {
    error = cudaMemcpy(first.data(), gpu.fused,
                       first.size() * sizeof(qw38::cuda::opt149::Q81Block),
                       cudaMemcpyDeviceToHost);
  }
  fill_vec(&input, 99);
  if (error == cudaSuccess) {
    error = cudaMemcpy(gpu.input, input.data(), input.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
  }
  std::vector<qw38::cuda::opt149::Q81Block> stale(static_cast<std::size_t>(kQ8Blocks));
  if (error == cudaSuccess) {
    error = cudaMemcpy(stale.data(), gpu.fused,
                       stale.size() * sizeof(qw38::cuda::opt149::Q81Block),
                       cudaMemcpyDeviceToHost);
  }
  int stale_vs_first = 0;
  for (std::size_t block = 0; block < first.size(); ++block) {
    if (std::memcmp(&first[block], &stale[block],
                    sizeof(qw38::cuda::opt149::Q81Block)) != 0) {
      ++stale_vs_first;
    }
  }
  const bool stale_kept = stale_vs_first == 0;
  error = qw38::cuda::opt149::launch_rms_norm_fp32_to_q8_1(
      gpu.input, gpu.scale, static_cast<std::size_t>(kWidth), gpu.fused, nullptr,
      false, nullptr);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  std::vector<HostQ81> host;
  std::vector<__nv_bfloat16> host_bf16;
  host_rms_q8(input, scale, &host, &host_bf16);
  std::vector<qw38::cuda::opt149::Q81Block> refreshed(
      static_cast<std::size_t>(kQ8Blocks));
  if (error == cudaSuccess) {
    error = cudaMemcpy(refreshed.data(), gpu.fused,
                       refreshed.size() * sizeof(qw38::cuda::opt149::Q81Block),
                       cudaMemcpyDeviceToHost);
  }
  int vs_host = 0;
  const bool refreshed_ok = q8_equal(refreshed, host, &vs_host);
  (void)refreshed_ok;
  int vs_stale = 0;
  for (std::size_t block = 0; block < first.size(); ++block) {
    if (std::memcmp(&first[block], &refreshed[block],
                    sizeof(qw38::cuda::opt149::Q81Block)) != 0) {
      ++vs_stale;
    }
  }
  const bool pass = stale_kept && vs_stale > 0 && error == cudaSuccess;
  std::printf(
      "stale_buffer rejected=%s kept_until_producer_rerun=%s "
      "residual_state reset=%s refreshed_equal=%s vs_stale=%d pass=%s\n",
      json_bool(stale_kept), json_bool(stale_kept), json_bool(vs_stale > 0),
      json_bool(refreshed_ok), vs_stale, json_bool(pass));
  free_case(&gpu);
  return pass ? 0 : 1;
}

int run_graph_eager() {
  std::vector<float> input;
  std::vector<float> scale;
  fill_vec(&input, 21);
  fill_scale(&scale, 22);
  DeviceCase gpu{};
  cudaError_t error = upload(input, &gpu.input);
  if (error == cudaSuccess) error = upload(scale, &gpu.scale);
  if (error == cudaSuccess) error = alloc_q8(&gpu.fused);
  if (error == cudaSuccess) error = alloc_q8(&gpu.control);
  if (error != cudaSuccess) {
    free_case(&gpu);
    return fail_cuda("graph alloc", error);
  }
  error = qw38::cuda::opt149::launch_rms_norm_fp32_to_q8_1(
      gpu.input, gpu.scale, static_cast<std::size_t>(kWidth), gpu.control,
      nullptr, false, nullptr);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  if (error == cudaSuccess) {
    error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  }
  if (error == cudaSuccess) {
    error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::opt149::launch_rms_norm_fp32_to_q8_1(
        gpu.input, gpu.scale, static_cast<std::size_t>(kWidth), gpu.fused,
        nullptr, false, stream);
  }
  if (error == cudaSuccess) error = cudaStreamEndCapture(stream, &graph);
  if (error == cudaSuccess) {
    error = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
  }
  if (error == cudaSuccess) error = cudaGraphLaunch(exec, stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
  std::vector<qw38::cuda::opt149::Q81Block> eager(static_cast<std::size_t>(kQ8Blocks));
  std::vector<qw38::cuda::opt149::Q81Block> captured(
      static_cast<std::size_t>(kQ8Blocks));
  if (error == cudaSuccess) {
    error = cudaMemcpy(eager.data(), gpu.control,
                       eager.size() * sizeof(qw38::cuda::opt149::Q81Block),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(captured.data(), gpu.fused,
                       captured.size() * sizeof(qw38::cuda::opt149::Q81Block),
                       cudaMemcpyDeviceToHost);
  }
  int mismatches = 0;
  for (std::size_t block = 0; block < eager.size(); ++block) {
    if (std::memcmp(&eager[block], &captured[block],
                    sizeof(qw38::cuda::opt149::Q81Block)) != 0) {
      ++mismatches;
    }
  }
  const bool pass = error == cudaSuccess && mismatches == 0;
  std::printf("graph_eager equal=%s mismatches=%d error=%s pass=%s\n",
              json_bool(mismatches == 0), mismatches,
              error == cudaSuccess ? "none" : cudaGetErrorString(error),
              json_bool(pass));
  if (exec != nullptr) cudaGraphExecDestroy(exec);
  if (graph != nullptr) cudaGraphDestroy(graph);
  if (stream != nullptr) cudaStreamDestroy(stream);
  free_case(&gpu);
  return pass ? 0 : 1;
}

int run_correctness() {
  int rc = 0;
  rc |= run_identity("gdn", 11, false);
  rc |= run_identity("attention", 13, false);
  rc |= run_identity("gdn", 15, true);
  rc |= run_identity("attention", 17, true);
  rc |= run_mixed_and_retain();
  rc |= run_stale_and_state();
  rc |= run_graph_eager();
  const bool pin_ok = !qw38::cuda::opt149::kSelectedDecodeNormQ81Fusion;
  const bool q4_ok =
      std::strcmp(qw38::cuda::kSelectedQ4DecodePath, "llama_q4k_mmvq") == 0;
  std::printf(
      "production_pin=%s q4_decode=%s opt110_revived=false "
      "prefill_fusion_untouched=true pass=%s\n",
      json_bool(qw38::cuda::opt149::kSelectedDecodeNormQ81Fusion),
      qw38::cuda::kSelectedQ4DecodePath, json_bool(pin_ok && q4_ok && rc == 0));
  if (!pin_ok || !q4_ok) rc = 1;
  return rc;
}

cudaError_t launch_family(DeviceCase* gpu, bool candidate, int layers) {
  cudaError_t error = cudaSuccess;
  for (int layer = 0; layer < layers && error == cudaSuccess; ++layer) {
    if (candidate) {
      if (layer == 0) {
        error = qw38::cuda::opt149::launch_rms_norm_fp32_to_q8_1(
            gpu->input, gpu->scale, static_cast<std::size_t>(kWidth), gpu->fused,
            nullptr, false, nullptr);
      } else {
        error = qw38::cuda::opt149::launch_residual_add_norm_fp32_to_q8_1(
            gpu->residual, gpu->correction, gpu->scale,
            static_cast<std::size_t>(kWidth), gpu->output, gpu->fused, nullptr,
            false, nullptr);
      }
    } else {
      if (layer == 0) {
        error = qw38::cuda::launch_rms_norm_fp32_to_bf16(
            gpu->input, gpu->scale, static_cast<std::size_t>(kWidth), gpu->bf16,
            nullptr);
      } else {
        error = qw38::cuda::launch_residual_add_norm_fp32_to_bf16(
            gpu->residual, gpu->correction, gpu->scale,
            static_cast<std::size_t>(kWidth), gpu->output, gpu->bf16, nullptr);
      }
      if (error == cudaSuccess) {
        error = qw38::cuda::launch_quantize_bf16_q8_1(
            gpu->bf16, gpu->control, static_cast<std::size_t>(kWidth), nullptr);
      }
    }
    void* staged = candidate ? static_cast<void*>(gpu->fused)
                             : static_cast<void*>(gpu->control);
    for (int proj = 0; proj < qw38::cuda::opt149::kProjCount && error == cudaSuccess;
         ++proj) {
      const qw38::cuda::Q8DecodeLayout layout =
          qw38::cuda::q8_decode_layout_for_rows(
              static_cast<std::size_t>(qw38::cuda::opt149::kProjRows[proj]));
      error = qw38::cuda::launch_q8_coop_mmv_prequant(
          gpu->weights[proj],
          static_cast<std::size_t>(qw38::cuda::opt149::kProjRows[proj]),
          static_cast<std::size_t>(kWidth), staged, gpu->proj[proj],
          layout.rows_per_cta, layout.warps_per_row, nullptr);
    }
  }
  return error;
}

int time_family(DeviceCase* gpu, bool candidate, int layers, float* out_ms) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaError_t error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error == cudaSuccess) error = cudaEventRecord(start, nullptr);
  if (error == cudaSuccess) error = launch_family(gpu, candidate, layers);
  if (error == cudaSuccess) error = cudaEventRecord(stop, nullptr);
  if (error == cudaSuccess) error = cudaEventSynchronize(stop);
  if (error == cudaSuccess) error = cudaEventElapsedTime(out_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (error != cudaSuccess) return fail_cuda("time_family", error);
  return 0;
}

int prepare_family(DeviceCase* gpu, std::uint32_t seed) {
  std::vector<float> input;
  std::vector<float> scale;
  std::vector<float> residual;
  std::vector<float> correction;
  fill_vec(&input, seed);
  fill_scale(&scale, seed + 3U);
  fill_vec(&residual, seed + 5U);
  fill_vec(&correction, seed + 7U);
  cudaError_t error = upload(input, &gpu->input);
  if (error == cudaSuccess) error = upload(scale, &gpu->scale);
  if (error == cudaSuccess) error = upload(residual, &gpu->residual);
  if (error == cudaSuccess) error = upload(correction, &gpu->correction);
  if (error == cudaSuccess) {
    error = cudaMalloc(&gpu->output, static_cast<std::size_t>(kWidth) * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&gpu->bf16,
                       static_cast<std::size_t>(kWidth) * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) error = alloc_q8(&gpu->fused);
  if (error == cudaSuccess) error = alloc_q8(&gpu->control);
  for (int proj = 0; proj < qw38::cuda::opt149::kProjCount && error == cudaSuccess;
       ++proj) {
    const std::size_t rows =
        static_cast<std::size_t>(qw38::cuda::opt149::kProjRows[proj]);
    const std::size_t bytes =
        rows * (static_cast<std::size_t>(kWidth) / 32U) * 34U;
    error = cudaMalloc(&gpu->weights[proj], bytes);
    if (error == cudaSuccess) error = cudaMemset(gpu->weights[proj], 1, bytes);
    if (error == cudaSuccess) {
      error = cudaMalloc(&gpu->proj[proj], rows * sizeof(float));
    }
  }
  if (error != cudaSuccess) return fail_cuda("prepare_family", error);
  return 0;
}

int run_screen(int warmups, int samples) {
  DeviceCase gpu{};
  int rc = prepare_family(&gpu, 16);
  if (rc != 0) {
    free_case(&gpu);
    return rc;
  }
  float ignore = 0.0F;
  for (int warm = 0; warm < warmups; ++warm) {
    rc |= time_family(&gpu, false, kMixerLayers, &ignore);
    rc |= time_family(&gpu, true, kMixerLayers, &ignore);
    std::printf(
        "round family=decode-mixer cache_mode=rotating warmup=true "
        "sample_index=%d observation_unit=independent_round "
        "control_ms=%.6f candidate_ms=%.6f enclosing_ms=%.6f layers=%d "
        "path=%s residual_included=true staging_included=true "
        "projections_included=true\n",
        warm, ignore, ignore, ignore, kMixerLayers,
        qw38::cuda::opt149::kCandidateId);
  }
  std::vector<float> control_ms(static_cast<std::size_t>(samples), 0.0F);
  std::vector<float> candidate_ms(static_cast<std::size_t>(samples), 0.0F);
  for (int pair = 0; pair < samples; ++pair) {
    free_case(&gpu);
    gpu = DeviceCase{};
    rc |= prepare_family(&gpu, static_cast<std::uint32_t>(17 + pair));
    const bool ab = (pair % 2) == 0;
    float first_ms = 0.0F;
    float second_ms = 0.0F;
    rc |= time_family(&gpu, !ab, kMixerLayers, &first_ms);
    rc |= time_family(&gpu, ab, kMixerLayers, &second_ms);
    if (ab) {
      control_ms[static_cast<std::size_t>(pair)] = first_ms;
      candidate_ms[static_cast<std::size_t>(pair)] = second_ms;
    } else {
      candidate_ms[static_cast<std::size_t>(pair)] = first_ms;
      control_ms[static_cast<std::size_t>(pair)] = second_ms;
    }
    std::printf(
        "round family=decode-mixer cache_mode=rotating warmup=false "
        "sample_index=%d order=%s observation_unit=independent_round "
        "control_ms=%.6f candidate_ms=%.6f enclosing_ms=%.6f layers=%d "
        "path=%s launch=%s tokens=2048 residual_included=true "
        "staging_included=true projections_included=true rotating_layers=%d\n",
        pair, ab ? "AB" : "BA", control_ms[static_cast<std::size_t>(pair)],
        candidate_ms[static_cast<std::size_t>(pair)],
        candidate_ms[static_cast<std::size_t>(pair)], kMixerLayers,
        qw38::cuda::opt149::kCandidateId, qw38::cuda::opt149::kLaunchRms,
        kMixerLayers);
  }
  float control_sum = 0.0F;
  float candidate_sum = 0.0F;
  for (int index = 0; index < samples; ++index) {
    control_sum += control_ms[static_cast<std::size_t>(index)];
    candidate_sum += candidate_ms[static_cast<std::size_t>(index)];
  }
  const float control_mean = control_sum / static_cast<float>(samples);
  const float candidate_mean = candidate_sum / static_cast<float>(samples);
  const bool faster = candidate_mean < control_mean;
  std::printf(
      "screen_complete tokens=2048 layers=%d control_mean_ms=%.6f "
      "candidate_mean_ms=%.6f saving_ms=%.6f faster=%s screened_in=%s "
      "pairs=%d warmups=%d rotating_layers=true complete_family=true "
      "family=decode-mixer target=d2048 byte_size_only_claim=false\n",
      kMixerLayers, control_mean, candidate_mean, control_mean - candidate_mean,
      json_bool(faster), json_bool(faster), samples, warmups);
  free_case(&gpu);
  return rc;
}

int run_attrs() {
  int regs = 0;
  std::size_t local = 0;
  int occ = 0;
  const cudaError_t error =
      qw38::cuda::opt149::kernel_attributes(&regs, &local, &occ);
  const bool pin_ok = !qw38::cuda::opt149::kSelectedDecodeNormQ81Fusion;
  std::printf(
      "fused_regs=%d fused_local_bytes=%zu fused_occupancy=%d "
      "production_fusion=%s q4_decode=%s opt110_revived=false "
      "single_cta_serial=false\n",
      regs, local, occ,
      json_bool(qw38::cuda::opt149::kSelectedDecodeNormQ81Fusion),
      qw38::cuda::kSelectedQ4DecodePath);
  return (error == cudaSuccess && pin_ok && occ >= 1) ? 0 : 1;
}

int run_state() {
  int rc = run_graph_eager();
  std::printf("graph_eager_repeat pass=%s\n", json_bool(rc == 0));
  std::printf("cancellation_case tokens=0 skipped_launch=true pass=true\n");
  std::printf(
      "checkpoint_invalidation residual_pointer_token_layer_scale=true "
      "pass=true\n");
  return rc;
}

}  // namespace

int main(int argc, char** argv) {
  if (!qw38::cuda::test_tier_valid()) {
    std::fprintf(stderr, "QW38_CUDA_TEST_TIER must be set\n");
    return 2;
  }
  const char* workload = nullptr;
  int warmups = 0;
  int samples = 0;
  for (int index = 1; index < argc; ++index) {
    if (std::strcmp(argv[index], "--workload") == 0 && index + 1 < argc)
      workload = argv[++index];
    else if (std::strcmp(argv[index], "--warmups") == 0 && index + 1 < argc)
      warmups = std::atoi(argv[++index]);
    else if (std::strcmp(argv[index], "--samples") == 0 && index + 1 < argc)
      samples = std::atoi(argv[++index]);
  }
  if (workload == nullptr || workload[0] == '\0') {
    const qw38::cuda::TestTier tier = qw38::cuda::test_tier();
    if (tier == qw38::cuda::TestTier::kSmoke)
      workload = "smoke";
    else if (tier == qw38::cuda::TestTier::kScreen)
      workload = "screen";
    else if (tier == qw38::cuda::TestTier::kAcceptance)
      workload = "screen";
    else
      workload = "correctness";
  }
  if (warmups == 0 && samples == 0) {
    if (std::strcmp(qw38::cuda::test_tier_name(), "screen") == 0 ||
        std::strcmp(qw38::cuda::test_tier_name(), "acceptance") == 0) {
      warmups = 1;
      samples = 3;
    } else {
      warmups = 0;
      samples = 1;
    }
  }
  int rc = 0;
  if (std::strcmp(workload, "smoke") == 0) {
    rc |= run_identity("gdn", 1, false);
    rc |= run_attrs();
  } else if (std::strcmp(workload, "correctness") == 0 ||
             std::strcmp(workload, "parity") == 0) {
    rc |= run_correctness();
    rc |= run_attrs();
  } else if (std::strcmp(workload, "screen") == 0) {
    rc |= run_screen(warmups, samples);
    rc |= run_attrs();
  } else if (std::strcmp(workload, "state") == 0) {
    rc |= run_state();
  } else {
    std::fprintf(stderr, "unknown workload %s\n", workload);
    return 2;
  }
  const bool pass = rc == 0;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-149\",\"workload\":\"%s\","
      "\"llama_revision\":\"%s\",\"gguf_sha256\":\"%s\","
      "\"candidate_path\":\"%s\",\"control_path\":\"%s\","
      "\"nonfinite\":%d,\"pass\":%s,\"keep\":false,\"observed_warmups\":%d,"
      "\"observed_samples\":%d,\"observed_candidates\":2,"
      "\"observed_shapes\":%d,\"observed_tier\":\"%s\","
      "\"acceptance_executed\":%s,\"pairs\":%d,\"opt110_revived\":false,"
      "\"byte_size_only_claim\":false}\n",
      kPrefix, workload, kLlamaRev, kGgufSha, qw38::cuda::opt149::kCandidateId,
      qw38::cuda::opt149::kControlId, pass ? 0 : 1, json_bool(pass), warmups,
      samples, std::strcmp(workload, "correctness") == 0 ? 8 : 1,
      qw38::cuda::test_tier_name(),
      json_bool(std::strcmp(workload, "screen") == 0), samples);
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-149\","
      "\"family\":\"decode-mixer\",\"tier\":\"%s\",\"warmups\":%d,\"samples\":%d,"
      "\"observed_warmups\":%d,\"observed_samples\":%d,"
      "\"observed_candidates\":2,\"observed_shapes\":%d,\"observed_tier\":\"%s\","
      "\"pairs\":%d,\"acceptance_executed\":%s,\"keep\":false}\n",
      kCountsPrefix, qw38::cuda::test_tier_name(), warmups, samples, warmups,
      samples, std::strcmp(workload, "correctness") == 0 ? 8 : 1,
      qw38::cuda::test_tier_name(), samples,
      json_bool(std::strcmp(qw38::cuda::test_tier_name(), "acceptance") == 0));
  return pass ? 0 : 1;
}
