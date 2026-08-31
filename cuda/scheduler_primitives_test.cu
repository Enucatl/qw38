#include "scheduler_primitives.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "quant.h"
#include "gdn_step.h"
#include "quant_mmv.h"

namespace {

int fail_cuda(const char* operation, cudaError_t error) {
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
  return 1;
}

float bf16_float(__nv_bfloat16 value) { return __bfloat162float(value); }

struct Metrics {
  float maximum = 0.0F;
  double squared = 0.0;
  std::size_t count = 0;
  std::size_t nonfinite = 0;
};

void measure(const std::vector<__nv_bfloat16>& actual,
             const std::vector<__nv_bfloat16>& expected, Metrics* metrics) {
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const float value = bf16_float(actual[index]);
    const float reference = bf16_float(expected[index]);
    if (!std::isfinite(value)) ++metrics->nonfinite;
    const float error = std::fabs(value - reference);
    metrics->maximum = std::max(metrics->maximum, error);
    metrics->squared += static_cast<double>(error) * error;
    ++metrics->count;
  }
}

int run_pointwise() {
  constexpr std::size_t kResidual = 5120;
  constexpr std::size_t kFfn = 17408;
  std::vector<__nv_bfloat16> residual(kResidual);
  std::vector<float> scale(kResidual);
  std::vector<float> correction(kResidual);
  std::vector<float> gate(kFfn);
  std::vector<float> up(kFfn);
  for (std::size_t index = 0; index < kResidual; ++index) {
    residual[index] = __float2bfloat16_rn(
        std::sin(static_cast<float>(index) * 0.013F) * 0.75F);
    scale[index] = 0.875F + static_cast<float>(index % 11) * 0.015625F;
    correction[index] =
        static_cast<float>(static_cast<int>(index % 17) - 8) * 0.0078125F;
  }
  for (std::size_t index = 0; index < kFfn; ++index) {
    gate[index] =
        static_cast<float>(static_cast<int>(index % 29) - 14) * 0.0625F;
    up[index] = std::cos(static_cast<float>(index) * 0.017F) * 0.5F;
  }
  float sum = 0.0F;
  for (auto item : residual) {
    const float value = bf16_float(item);
    sum += value * value;
  }
  const float inverse = 1.0F / std::sqrt(sum / kResidual + 1.0e-6F);
  std::vector<__nv_bfloat16> expected_norm(kResidual);
  std::vector<__nv_bfloat16> expected_residual(kResidual);
  std::vector<__nv_bfloat16> expected_swiglu(kFfn);
  for (std::size_t index = 0; index < kResidual; ++index) {
    expected_norm[index] = __float2bfloat16_rn(
        bf16_float(residual[index]) * inverse * scale[index]);
    expected_residual[index] = __float2bfloat16_rn(
        bf16_float(residual[index]) + correction[index]);
  }
  for (std::size_t index = 0; index < kFfn; ++index) {
    expected_swiglu[index] = __float2bfloat16_rn(
        gate[index] / (1.0F + std::exp(-gate[index])) * up[index]);
  }
  __nv_bfloat16* device_residual = nullptr;
  __nv_bfloat16* device_norm = nullptr;
  __nv_bfloat16* device_added = nullptr;
  __nv_bfloat16* device_swiglu = nullptr;
  float* device_scale = nullptr;
  float* device_correction = nullptr;
  float* device_gate = nullptr;
  float* device_up = nullptr;
  cudaError_t error = cudaMalloc(&device_residual, residual.size() * 2);
#define QW38_ALLOC(pointer, count)                                            \
  if (error == cudaSuccess)                                                   \
  error = cudaMalloc(&(pointer), (count) * sizeof(*(pointer)))
  QW38_ALLOC(device_norm, kResidual);
  QW38_ALLOC(device_added, kResidual);
  QW38_ALLOC(device_swiglu, kFfn);
  QW38_ALLOC(device_scale, kResidual);
  QW38_ALLOC(device_correction, kResidual);
  QW38_ALLOC(device_gate, kFfn);
  QW38_ALLOC(device_up, kFfn);
#undef QW38_ALLOC
  if (error != cudaSuccess) return fail_cuda("pointwise cudaMalloc", error);
#define QW38_COPY(pointer, source)                                            \
  if (error == cudaSuccess)                                                   \
  error = cudaMemcpy((pointer), (source).data(),                              \
                     (source).size() * sizeof((source)[0]), cudaMemcpyHostToDevice)
  QW38_COPY(device_residual, residual);
  QW38_COPY(device_scale, scale);
  QW38_COPY(device_correction, correction);
  QW38_COPY(device_gate, gate);
  QW38_COPY(device_up, up);
#undef QW38_COPY
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_rms_norm_bf16(
        device_residual, device_scale, kResidual, device_norm, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_residual_add_bf16(
        device_residual, device_correction, kResidual, device_added, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_swiglu_bf16(
        device_gate, device_up, kFfn, device_swiglu, nullptr);
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (error == cudaSuccess) error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  float total_ms = 0.0F;
  for (int sample = -3; error == cudaSuccess && sample < 30; ++sample) {
    error = cudaEventRecord(start);
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_rms_norm_bf16(
          device_residual, device_scale, kResidual, device_norm, nullptr);
    }
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_residual_add_bf16(
          device_residual, device_correction, kResidual, device_added, nullptr);
    }
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_swiglu_bf16(
          device_gate, device_up, kFfn, device_swiglu, nullptr);
    }
    if (error == cudaSuccess) error = cudaEventRecord(stop);
    if (error == cudaSuccess) error = cudaEventSynchronize(stop);
    float elapsed = 0.0F;
    if (error == cudaSuccess) error = cudaEventElapsedTime(&elapsed, start, stop);
    if (sample >= 0) total_ms += elapsed;
  }
  std::vector<__nv_bfloat16> actual_norm(kResidual);
  std::vector<__nv_bfloat16> actual_added(kResidual);
  std::vector<__nv_bfloat16> actual_swiglu(kFfn);
#define QW38_READ(destination, pointer)                                       \
  if (error == cudaSuccess)                                                   \
  error = cudaMemcpy((destination).data(), (pointer),                         \
                     (destination).size() * sizeof((destination)[0]),         \
                     cudaMemcpyDeviceToHost)
  QW38_READ(actual_norm, device_norm);
  QW38_READ(actual_added, device_added);
  QW38_READ(actual_swiglu, device_swiglu);
#undef QW38_READ
  if (error != cudaSuccess) return fail_cuda("pointwise execution", error);
  Metrics metrics;
  measure(actual_norm, expected_norm, &metrics);
  measure(actual_added, expected_residual, &metrics);
  measure(actual_swiglu, expected_swiglu, &metrics);
  const float rms = static_cast<float>(
      std::sqrt(metrics.squared / static_cast<double>(metrics.count)));
  std::printf("scheduler_pointwise=bf16 max_abs=%.9g rms=%.9g nonfinite=%zu "
              "mean_ms=%.9g\n",
              metrics.maximum, rms, metrics.nonfinite, total_ms / 30.0F);
  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  cudaFree(device_up);
  cudaFree(device_gate);
  cudaFree(device_correction);
  cudaFree(device_scale);
  cudaFree(device_swiglu);
  cudaFree(device_added);
  cudaFree(device_norm);
  cudaFree(device_residual);
  return metrics.nonfinite == 0 && metrics.maximum <= 0.0078125F &&
                 rms <= 0.0005F
             ? 0
             : 1;
}

int run_layouts() {
  constexpr std::size_t kQueryValues = 24 * 256;
  constexpr std::size_t kGdnHeads = 48;
  constexpr std::size_t kGdnWidth = 128;
  std::vector<float> packed(kQueryValues * 2);
  for (std::size_t index = 0; index < packed.size(); ++index) {
    packed[index] = static_cast<float>(index);
  }
  std::vector<float> query(kQueryValues);
  std::vector<float> gate(kQueryValues);
  std::vector<float> alpha(kGdnHeads);
  std::vector<float> beta(kGdnHeads);
  std::vector<float> folded(kGdnHeads);
  std::vector<float> bias(kGdnHeads);
  std::vector<float> recurrent(kGdnHeads * kGdnWidth);
  std::vector<float> gate_tiled(kGdnHeads * kGdnWidth);
  std::vector<float> norm(kGdnWidth);
  for (std::size_t index = 0; index < kGdnHeads; ++index) {
    alpha[index] = static_cast<float>(static_cast<int>(index % 9) - 4) * 0.125F;
    beta[index] = static_cast<float>(static_cast<int>(index % 7) - 3) * 0.25F;
    folded[index] = -0.25F - static_cast<float>(index % 5) * 0.03125F;
    bias[index] = static_cast<float>(index % 3) * 0.0625F;
  }
  for (std::size_t index = 0; index < recurrent.size(); ++index) {
    recurrent[index] = std::sin(static_cast<float>(index) * 0.009F) * 0.5F;
    gate_tiled[index] = std::cos(static_cast<float>(index) * 0.007F);
  }
  for (std::size_t index = 0; index < norm.size(); ++index) {
    norm[index] = 0.875F + static_cast<float>(index % 9) * 0.015625F;
  }
  float* d_packed = nullptr;
  float* d_query = nullptr;
  float* d_gate = nullptr;
  float* d_alpha = nullptr;
  float* d_beta = nullptr;
  float* d_folded = nullptr;
  float* d_bias = nullptr;
  float* d_decay = nullptr;
  float* d_update = nullptr;
  float* d_recurrent = nullptr;
  float* d_gate_tiled = nullptr;
  float* d_norm = nullptr;
  __nv_bfloat16* d_output = nullptr;
  cudaError_t error = cudaMalloc(&d_packed, packed.size() * sizeof(float));
#define QW38_ALLOC(pointer, count)                                            \
  if (error == cudaSuccess)                                                   \
  error = cudaMalloc(&(pointer), (count) * sizeof(*(pointer)))
  QW38_ALLOC(d_query, query.size());
  QW38_ALLOC(d_gate, gate.size());
  QW38_ALLOC(d_alpha, alpha.size());
  QW38_ALLOC(d_beta, beta.size());
  QW38_ALLOC(d_folded, folded.size());
  QW38_ALLOC(d_bias, bias.size());
  QW38_ALLOC(d_decay, alpha.size());
  QW38_ALLOC(d_update, beta.size());
  QW38_ALLOC(d_recurrent, recurrent.size());
  QW38_ALLOC(d_gate_tiled, gate_tiled.size());
  QW38_ALLOC(d_norm, norm.size());
  QW38_ALLOC(d_output, recurrent.size());
#undef QW38_ALLOC
#define QW38_COPY(pointer, source)                                            \
  if (error == cudaSuccess)                                                   \
  error = cudaMemcpy((pointer), (source).data(),                              \
                     (source).size() * sizeof((source)[0]), cudaMemcpyHostToDevice)
  QW38_COPY(d_packed, packed);
  QW38_COPY(d_alpha, alpha);
  QW38_COPY(d_beta, beta);
  QW38_COPY(d_folded, folded);
  QW38_COPY(d_bias, bias);
  QW38_COPY(d_recurrent, recurrent);
  QW38_COPY(d_gate_tiled, gate_tiled);
  QW38_COPY(d_norm, norm);
#undef QW38_COPY
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_split_attention_query_gate(
        d_packed, 24, 256, d_query, d_gate, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_prepare_gdn_gates(
        d_alpha, d_beta, d_folded, d_bias, 16, 3, d_decay, d_update, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_gdn_gated_output(
        d_recurrent, d_gate_tiled, d_norm, 16, 3, 128, d_output, nullptr);
  }
  std::vector<float> decay(kGdnHeads);
  std::vector<float> update(kGdnHeads);
  std::vector<__nv_bfloat16> output(recurrent.size());
  if (error == cudaSuccess) {
    error = cudaMemcpy(query.data(), d_query, query.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(gate.data(), d_gate, gate.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(decay.data(), d_decay, decay.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(update.data(), d_update, update.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(output.data(), d_output, output.size() * sizeof(output[0]),
                       cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) return fail_cuda("layout execution", error);
  bool split_exact = true;
  for (std::size_t head = 0; head < 24; ++head) {
    split_exact = split_exact && query[head * 256] == packed[head * 512] &&
                  gate[head * 256 + 255] == packed[head * 512 + 511];
  }
  float maximum = 0.0F;
  double squared = 0.0;
  std::size_t compared = 0;
  for (std::size_t grouped = 0; grouped < kGdnHeads; ++grouped) {
    const std::size_t key_head = grouped / 3;
    const std::size_t replica = grouped % 3;
    const std::size_t tiled = replica * 16 + key_head;
    const float x = alpha[tiled] + bias[tiled];
    const float expected_decay = folded[tiled] * std::log1p(std::exp(x));
    const float expected_update = 1.0F / (1.0F + std::exp(-beta[tiled]));
    for (float pair : {decay[grouped] - expected_decay,
                       update[grouped] - expected_update}) {
      maximum = std::max(maximum, std::fabs(pair));
      squared += static_cast<double>(pair) * pair;
      ++compared;
    }
  }
  const float rms =
      static_cast<float>(std::sqrt(squared / static_cast<double>(compared)));
  std::size_t nonfinite = 0;
  for (auto item : output) {
    if (!std::isfinite(bf16_float(item))) ++nonfinite;
  }
  std::printf("scheduler_layout=production split_exact=%s max_abs=%.9g "
              "rms=%.9g nonfinite=%zu\n",
              split_exact ? "true" : "false", maximum, rms, nonfinite);
  cudaFree(d_output);
  cudaFree(d_norm);
  cudaFree(d_gate_tiled);
  cudaFree(d_recurrent);
  cudaFree(d_update);
  cudaFree(d_decay);
  cudaFree(d_bias);
  cudaFree(d_folded);
  cudaFree(d_beta);
  cudaFree(d_alpha);
  cudaFree(d_gate);
  cudaFree(d_query);
  cudaFree(d_packed);
  return split_exact && maximum <= 5.0e-7F && rms <= 1.0e-7F &&
                 nonfinite == 0
             ? 0
             : 1;
}

int run_row_decode() {
  std::vector<std::uint8_t> weights(3 * 144, 0);
  for (std::size_t row = 0; row < 3; ++row) {
    weights[row * 144] = 0x00;
    weights[row * 144 + 1] = 0x24;
    weights[row * 144 + 2] = 0x00;
    weights[row * 144 + 3] = 0x1C;
    for (std::size_t index = 4; index < 144; ++index) {
      weights[row * 144 + index] =
          static_cast<std::uint8_t>((index * 17 + row * 13) & 0xFFU);
    }
  }
  std::vector<float> decoded(256);
  if (!qw38::internal::decode_q4_k(weights.data() + 144, 144, decoded.data(),
                                   decoded.size()).is_ok()) return 1;
  std::vector<__nv_bfloat16> expected(256);
  for (std::size_t index = 0; index < 256; ++index) {
    expected[index] = __float2bfloat16_rn(decoded[index]);
  }
  std::uint8_t* d_weights = nullptr;
  __nv_bfloat16* d_output = nullptr;
  cudaError_t error = cudaMalloc(&d_weights, weights.size());
  if (error == cudaSuccess) error = cudaMalloc(&d_output, expected.size() * 2);
  if (error == cudaSuccess) {
    error = cudaMemcpy(d_weights, weights.data(), weights.size(),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quant_row_decode(
        qw38::cuda::QuantKind::kQ4K, d_weights, 3, 256, 1, d_output, nullptr);
  }
  std::vector<__nv_bfloat16> actual(256);
  if (error == cudaSuccess) {
    error = cudaMemcpy(actual.data(), d_output, actual.size() * 2,
                       cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) return fail_cuda("row decode", error);
  const bool exact = std::memcmp(actual.data(), expected.data(), 512) == 0;
  std::printf("scheduler_embedding=q4_k bf16_exact=%s\n",
              exact ? "true" : "false");
  cudaFree(d_output);
  cudaFree(d_weights);
  return exact ? 0 : 1;
}

int run_tiled_gdn() {
  const qw38::cuda::GdnConfig config{2, 6, 8, 8, 1};
  const std::size_t channels = qw38::cuda::gdn_convolution_channels(config);
  const std::size_t convolution_values =
      qw38::cuda::gdn_convolution_values(config);
  const std::size_t recurrent_values =
      qw38::cuda::gdn_recurrent_values(config);
  const std::size_t output_values = qw38::cuda::gdn_output_values(config);
  std::vector<float> grouped(channels);
  std::vector<float> tiled(channels);
  for (std::size_t index = 0; index < 32; ++index) {
    grouped[index] = std::sin(static_cast<float>(index) * 0.07F);
    tiled[index] = grouped[index];
  }
  for (std::size_t grouped_head = 0; grouped_head < 6; ++grouped_head) {
    const std::size_t key = grouped_head / 3;
    const std::size_t replica = grouped_head % 3;
    const std::size_t tiled_head = replica * 2 + key;
    for (std::size_t lane = 0; lane < 8; ++lane) {
      const float value = static_cast<float>(grouped_head * 8 + lane) * 0.015625F;
      grouped[32 + grouped_head * 8 + lane] = value;
      tiled[32 + tiled_head * 8 + lane] = value;
    }
  }
  std::vector<float> weights(convolution_values, 1.0F);
  std::vector<float> decay(6, -0.125F);
  std::vector<float> beta(6, 0.25F);
  float* d_grouped = nullptr;
  float* d_tiled = nullptr;
  float* d_weights = nullptr;
  float* d_decay = nullptr;
  float* d_beta = nullptr;
  float* d_committed_conv = nullptr;
  float* d_committed_rec = nullptr;
  float* d_grouped_conv = nullptr;
  float* d_grouped_rec = nullptr;
  float* d_tiled_conv = nullptr;
  float* d_tiled_rec = nullptr;
  float* d_conv_output = nullptr;
  float* d_grouped_output = nullptr;
  float* d_tiled_output = nullptr;
  cudaError_t error = cudaMalloc(&d_grouped, channels * sizeof(float));
#define QW38_ALLOC(pointer, count)                                            \
  if (error == cudaSuccess)                                                   \
  error = cudaMalloc(&(pointer), (count) * sizeof(*(pointer)))
  QW38_ALLOC(d_tiled, channels);
  QW38_ALLOC(d_weights, convolution_values);
  QW38_ALLOC(d_decay, decay.size());
  QW38_ALLOC(d_beta, beta.size());
  QW38_ALLOC(d_committed_conv, convolution_values);
  QW38_ALLOC(d_committed_rec, recurrent_values);
  QW38_ALLOC(d_grouped_conv, convolution_values);
  QW38_ALLOC(d_grouped_rec, recurrent_values);
  QW38_ALLOC(d_tiled_conv, convolution_values);
  QW38_ALLOC(d_tiled_rec, recurrent_values);
  QW38_ALLOC(d_conv_output, channels);
  QW38_ALLOC(d_grouped_output, output_values);
  QW38_ALLOC(d_tiled_output, output_values);
#undef QW38_ALLOC
#define QW38_COPY(pointer, source)                                            \
  if (error == cudaSuccess)                                                   \
  error = cudaMemcpy((pointer), (source).data(),                              \
                     (source).size() * sizeof((source)[0]), cudaMemcpyHostToDevice)
  QW38_COPY(d_grouped, grouped);
  QW38_COPY(d_tiled, tiled);
  QW38_COPY(d_weights, weights);
  QW38_COPY(d_decay, decay);
  QW38_COPY(d_beta, beta);
#undef QW38_COPY
  if (error == cudaSuccess) {
    error = cudaMemset(d_committed_conv, 0, convolution_values * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMemset(d_committed_rec, 0, recurrent_values * sizeof(float));
  }
  const qw38::cuda::GdnState committed{d_committed_conv, d_committed_rec};
  const qw38::cuda::GdnState grouped_candidate{d_grouped_conv, d_grouped_rec};
  const qw38::cuda::GdnState tiled_candidate{d_tiled_conv, d_tiled_rec};
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_gdn_prepare(
        config, d_grouped, d_weights, d_decay, d_beta, committed,
        grouped_candidate, d_conv_output, d_grouped_output, nullptr);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_gdn_prepare_tiled(
        config, d_tiled, d_weights, d_decay, d_beta, committed, tiled_candidate,
        d_conv_output, d_tiled_output, nullptr);
  }
  std::vector<float> grouped_output(output_values);
  std::vector<float> tiled_output(output_values);
  std::vector<float> grouped_state(recurrent_values);
  std::vector<float> tiled_state(recurrent_values);
  if (error == cudaSuccess) {
    error = cudaMemcpy(grouped_output.data(), d_grouped_output,
                       output_values * sizeof(float), cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(tiled_output.data(), d_tiled_output,
                       output_values * sizeof(float), cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(grouped_state.data(), d_grouped_rec,
                       recurrent_values * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(tiled_state.data(), d_tiled_rec,
                       recurrent_values * sizeof(float),
                       cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) return fail_cuda("tiled GDN", error);
  const bool exact =
      grouped_output == tiled_output && grouped_state == tiled_state;
  std::printf("scheduler_gdn=tiled_to_grouped exact=%s\n",
              exact ? "true" : "false");
  cudaFree(d_tiled_output);
  cudaFree(d_grouped_output);
  cudaFree(d_conv_output);
  cudaFree(d_tiled_rec);
  cudaFree(d_tiled_conv);
  cudaFree(d_grouped_rec);
  cudaFree(d_grouped_conv);
  cudaFree(d_committed_rec);
  cudaFree(d_committed_conv);
  cudaFree(d_beta);
  cudaFree(d_decay);
  cudaFree(d_weights);
  cudaFree(d_tiled);
  cudaFree(d_grouped);
  return exact ? 0 : 1;
}

}  // namespace

int main() {
  if (run_pointwise() != 0 || run_layouts() != 0 || run_row_decode() != 0 ||
      run_tiled_gdn() != 0) {
    return 1;
  }
  std::printf("status=passed\n");
  return 0;
}
