#include "quant_mmv.h"

#include "kernel_parity.cuh"
#include "quant.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

#include <cuda_fp16.h>

namespace {

constexpr char kPrefix[] = "QW38_FFN_TILE_AB_RESULT=";
constexpr int kWarmups = 3;
constexpr int kMeasured = 30;
constexpr std::size_t kEnvelopePrompt = 64;
constexpr std::size_t kEnvelopeRows = 256;
constexpr std::size_t kEnvelopeCols = 256;

struct Tile {
  const char* id;
  unsigned quality_i;
  unsigned prompt_tile;
};

constexpr Tile kTiles[] = {
    {"i64_j32", 64, 32},   {"i64_j64", 64, 64},   {"i64_j128", 64, 128},
    {"i128_j32", 128, 32}, {"i128_j64", 128, 64}, {"i128_j128", 128, 128},
};
constexpr int kTileCount = 6;
constexpr int kBaseline = 5;

struct Projection {
  const char* id;
  std::size_t rows;
  std::size_t columns;
};

constexpr Projection kProjections[] = {
    {"gate", 17408, 5120},
    {"up", 17408, 5120},
    {"down", 5120, 17408},
};
constexpr int kProjectionCount = 3;

int fail_cuda(const char* operation, cudaError_t error) {
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
  return 1;
}

const char* json_bool(bool value) { return value ? "true" : "false"; }

void print_array(FILE* file, const char* key, const std::vector<float>& values) {
  std::fprintf(file, "\"%s\":[", key);
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) std::fprintf(file, ",");
    std::fprintf(file, "%.9g", static_cast<double>(values[index]));
  }
  std::fprintf(file, "]");
}

void write_u16(std::uint8_t* bytes, std::uint16_t value) {
  bytes[0] = static_cast<std::uint8_t>(value & 0xFFU);
  bytes[1] = static_cast<std::uint8_t>(value >> 8U);
}

void fill_weights(std::size_t rows, std::size_t columns,
                  std::vector<std::uint8_t>* weights) {
  constexpr std::size_t kBlockBytes = 144;
  weights->assign(rows * (columns / 256) * kBlockBytes, 0);
  for (std::size_t index = 0; index < weights->size(); ++index) {
    (*weights)[index] = static_cast<std::uint8_t>((index * 73 + 19) & 0xFFU);
  }
  for (std::size_t offset = 0; offset < weights->size(); offset += kBlockBytes) {
    write_u16(weights->data() + offset, 0x2400U);
    write_u16(weights->data() + offset + 2, 0x1C00U);
  }
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

void fill_prompt(std::size_t prompt_rows, std::size_t columns,
                 std::vector<__nv_bfloat16>* prompt) {
  prompt->resize(prompt_rows * columns);
  for (std::size_t index = 0; index < prompt->size(); ++index) {
    (*prompt)[index] = __float2bfloat16_rn(
        unit_normal(static_cast<std::uint32_t>(index), 0xA11CE5u));
  }
}

bool reference_dequant_gemm(const std::vector<std::uint8_t>& weights,
                            std::size_t output_rows, std::size_t columns,
                            const std::vector<__nv_bfloat16>& prompt,
                            std::size_t prompt_rows,
                            std::vector<float>* output) {
  output->assign(prompt_rows * output_rows, 0.0F);
  std::vector<float> decoded(256);
  for (std::size_t out = 0; out < output_rows; ++out) {
    for (std::size_t prompt_row = 0; prompt_row < prompt_rows; ++prompt_row) {
      float sum = 0.0F;
      for (std::size_t block = 0; block < columns / 256; ++block) {
        const std::uint8_t* packed =
            weights.data() + (out * (columns / 256) + block) * 144;
        const qw38::Status status = qw38::internal::decode_q4_k(
            packed, 144, decoded.data(), decoded.size());
        if (!status.is_ok()) return false;
        for (std::size_t within = 0; within < 256; ++within) {
          const std::size_t column = block * 256 + within;
          sum += decoded[within] *
                 __bfloat162float(prompt[prompt_row * columns + column]);
        }
      }
      (*output)[prompt_row * output_rows + out] = sum;
    }
  }
  return true;
}

bool ds4_q4k_association_ok(const std::vector<float>& got,
                            const std::vector<float>& ref, std::size_t columns,
                            float* max_abs, float* rms, std::size_t* nonfinite,
                            std::size_t* bad) {
  const auto diagnostics = qw38::cuda::kernel_parity::check_close(
      got, ref, qw38::cuda::kernel_parity::Family::Q4_K, columns,
      qw38::cuda::kernel_parity::Class::QuantizedOperationAssociation, false);
  *max_abs = diagnostics.max_abs;
  *rms = diagnostics.rms;
  *nonfinite = diagnostics.nonfinite_count;
  *bad = diagnostics.failing_count;
  return diagnostics.pass;
}

std::size_t count_nonfinite(const std::vector<float>& values) {
  std::size_t count = 0;
  for (float value : values) {
    if (!std::isfinite(value)) ++count;
  }
  return count;
}

struct Envelope {
  float max_abs = 0.0F;
  float rms = 0.0F;
  std::size_t nonfinite = 0;
  std::size_t bad = 0;
  bool ok = false;
};

struct Candidate {
  std::vector<float> warmup;
  std::vector<float> samples;
  float mean_ms = 0.0F;
  int occupancy = 0;
  bool launch_ok = false;
  bool eligible = false;
  Envelope envelope;
  std::size_t timed_nonfinite = 0;
};

struct ProjectionResult {
  Candidate candidates[kTileCount];
  int winner = kBaseline;
  bool win = false;
};

}  // namespace

int main(int argc, char** argv) {
  const char* raw_path =
      argc > 1 ? argv[1]
               : "evidence/optimization/opt037-ffn-tiles/ffn-tile-ab-raw.txt";
  FILE* raw = std::fopen(raw_path, "w");
  if (raw == nullptr) {
    std::fprintf(stderr, "failed to open %s\n", raw_path);
    return 1;
  }

  cudaDeviceProp props{};
  if (cudaGetDeviceProperties(&props, 0) != cudaSuccess) {
    std::fprintf(stderr, "cudaGetDeviceProperties failed\n");
    std::fclose(raw);
    return 1;
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaError_t error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error != cudaSuccess) {
    std::fclose(raw);
    return fail_cuda("cudaEventCreate", error);
  }

  ProjectionResult results[kProjectionCount];

  std::vector<std::uint8_t> env_weights;
  fill_weights(kEnvelopeRows, kEnvelopeCols, &env_weights);
  std::vector<__nv_bfloat16> env_prompt;
  fill_prompt(kEnvelopePrompt, kEnvelopeCols, &env_prompt);
  std::vector<float> env_expected;
  if (!reference_dequant_gemm(env_weights, kEnvelopeRows, kEnvelopeCols,
                              env_prompt, kEnvelopePrompt, &env_expected)) {
    std::fprintf(stderr, "envelope host GEMM failed\n");
    std::fclose(raw);
    return 1;
  }

  std::uint8_t* env_w = nullptr;
  __nv_bfloat16* env_p = nullptr;
  qw38::cuda::Q8Block* env_y = nullptr;
  float* env_out = nullptr;
  error = cudaMalloc(&env_w, env_weights.size());
  if (error == cudaSuccess) {
    error = cudaMalloc(&env_p, env_prompt.size() * sizeof(env_prompt[0]));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&env_y, qw38::cuda::q8_prompt_workspace_bytes(
                                   kEnvelopePrompt, kEnvelopeCols));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&env_out, env_expected.size() * sizeof(float));
  }
  if (error != cudaSuccess) {
    std::fclose(raw);
    return fail_cuda("envelope cudaMalloc", error);
  }
  error = cudaMemcpy(env_w, env_weights.data(), env_weights.size(),
                     cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(env_p, env_prompt.data(),
                       env_prompt.size() * sizeof(env_prompt[0]),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_mmq_q8_1(
        qw38::cuda::QuantKind::kQ4K, env_p, kEnvelopePrompt, kEnvelopeCols,
        env_y, nullptr);
  }
  if (error != cudaSuccess) {
    std::fclose(raw);
    return fail_cuda("envelope quantize", error);
  }

  Envelope envelopes[kTileCount];
  for (int tile = 0; tile < kTileCount; ++tile) {
    error = qw38::cuda::launch_quant_mmq_mma_y_ij(
        qw38::cuda::QuantKind::kQ4K, env_w, kEnvelopeRows, kEnvelopeCols, env_y,
        kEnvelopePrompt, env_out, kTiles[tile].quality_i,
        kTiles[tile].prompt_tile, nullptr);
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    envelopes[tile].ok = false;
    if (error != cudaSuccess) {
      std::fprintf(raw, "envelope tile=%s launch=%s\n", kTiles[tile].id,
                   cudaGetErrorString(error));
      continue;
    }
    std::vector<float> actual(env_expected.size());
    error = cudaMemcpy(actual.data(), env_out, actual.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
      std::fclose(raw);
      return fail_cuda("envelope D2H", error);
    }
    envelopes[tile].ok = ds4_q4k_association_ok(
        actual, env_expected, kEnvelopeCols, &envelopes[tile].max_abs,
        &envelopes[tile].rms, &envelopes[tile].nonfinite, &envelopes[tile].bad);
    std::fprintf(raw,
                 "envelope tile=%s ok=%s max_abs=%.9g rms=%.9g nonfinite=%zu "
                 "bad=%zu occupancy=%d\n",
                 kTiles[tile].id, envelopes[tile].ok ? "true" : "false",
                 envelopes[tile].max_abs, envelopes[tile].rms,
                 envelopes[tile].nonfinite, envelopes[tile].bad,
                 qw38::cuda::mma_mmq_occupancy_ij(qw38::cuda::QuantKind::kQ4K,
                                                  kTiles[tile].prompt_tile,
                                                  kTiles[tile].quality_i));
  }
  cudaFree(env_out);
  cudaFree(env_y);
  cudaFree(env_p);
  cudaFree(env_w);

  for (int proj = 0; proj < kProjectionCount; ++proj) {
    const Projection& shape = kProjections[proj];
    std::vector<std::uint8_t> weights;
    fill_weights(shape.rows, shape.columns, &weights);
    std::vector<__nv_bfloat16> prompt;
    fill_prompt(4096, shape.columns, &prompt);
    std::uint8_t* device_w = nullptr;
    __nv_bfloat16* device_p = nullptr;
    qw38::cuda::Q8Block* device_y = nullptr;
    float* device_out = nullptr;
    error = cudaMalloc(&device_w, weights.size());
    if (error == cudaSuccess) {
      error = cudaMalloc(&device_p, prompt.size() * sizeof(prompt[0]));
    }
    if (error == cudaSuccess) {
      error = cudaMalloc(&device_y, qw38::cuda::q8_prompt_workspace_bytes(
                                        4096, shape.columns));
    }
    if (error == cudaSuccess) {
      error = cudaMalloc(&device_out, 4096 * shape.rows * sizeof(float));
    }
    if (error != cudaSuccess) {
      std::fclose(raw);
      return fail_cuda("projection cudaMalloc", error);
    }
    error = cudaMemcpy(device_w, weights.data(), weights.size(),
                       cudaMemcpyHostToDevice);
    if (error == cudaSuccess) {
      error = cudaMemcpy(device_p, prompt.data(),
                         prompt.size() * sizeof(prompt[0]),
                         cudaMemcpyHostToDevice);
    }
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_quantize_mmq_q8_1(
          qw38::cuda::QuantKind::kQ4K, device_p, 4096, shape.columns, device_y,
          nullptr);
    }
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
      std::fclose(raw);
      return fail_cuda("projection quantize", error);
    }

    for (int tile = 0; tile < kTileCount; ++tile) {
      Candidate& slot = results[proj].candidates[tile];
      slot.occupancy = qw38::cuda::mma_mmq_occupancy_ij(
          qw38::cuda::QuantKind::kQ4K, kTiles[tile].prompt_tile,
          kTiles[tile].quality_i);
      slot.envelope = envelopes[tile];
      slot.launch_ok = true;
    }

    for (int round = 0; round < kWarmups + kMeasured; ++round) {
      for (int tile = 0; tile < kTileCount; ++tile) {
        Candidate& slot = results[proj].candidates[tile];
        if (!slot.launch_ok) continue;
        error = cudaEventRecord(start);
        if (error == cudaSuccess) {
          error = qw38::cuda::launch_quant_mmq_mma_y_ij(
              qw38::cuda::QuantKind::kQ4K, device_w, shape.rows, shape.columns,
              device_y, 4096, device_out, kTiles[tile].quality_i,
              kTiles[tile].prompt_tile, nullptr);
        }
        if (error == cudaSuccess) error = cudaEventRecord(stop);
        if (error == cudaSuccess) error = cudaEventSynchronize(stop);
        float milliseconds = 0.0F;
        if (error == cudaSuccess) {
          error = cudaEventElapsedTime(&milliseconds, start, stop);
        }
        if (error != cudaSuccess) {
          slot.launch_ok = false;
          std::fprintf(raw, "proj=%s tile=%s launch=%s\n", shape.id,
                       kTiles[tile].id, cudaGetErrorString(error));
          continue;
        }
        std::fprintf(raw, "proj=%s tile=%s round=%d ms=%.9g\n", shape.id,
                     kTiles[tile].id, round, milliseconds);
        if (round < kWarmups) {
          slot.warmup.push_back(milliseconds);
        } else {
          slot.samples.push_back(milliseconds);
        }
      }
    }

    for (int tile = 0; tile < kTileCount; ++tile) {
      Candidate& slot = results[proj].candidates[tile];
      slot.timed_nonfinite = 1;
      if (!slot.launch_ok) continue;
      error = qw38::cuda::launch_quant_mmq_mma_y_ij(
          qw38::cuda::QuantKind::kQ4K, device_w, shape.rows, shape.columns,
          device_y, 4096, device_out, kTiles[tile].quality_i,
          kTiles[tile].prompt_tile, nullptr);
      if (error == cudaSuccess) error = cudaDeviceSynchronize();
      if (error != cudaSuccess) {
        slot.launch_ok = false;
        continue;
      }
      std::vector<float> timed(4096 * shape.rows);
      error = cudaMemcpy(timed.data(), device_out, timed.size() * sizeof(float),
                         cudaMemcpyDeviceToHost);
      if (error != cudaSuccess) {
        std::fclose(raw);
        return fail_cuda("timed D2H", error);
      }
      slot.timed_nonfinite = count_nonfinite(timed);
    }
    float best = 1.0e30F;
    int winner = kBaseline;
    for (int tile = 0; tile < kTileCount; ++tile) {
      Candidate& slot = results[proj].candidates[tile];
      double sum = 0.0;
      for (float sample : slot.samples) sum += static_cast<double>(sample);
      slot.mean_ms =
          slot.samples.empty()
              ? 0.0F
              : static_cast<float>(sum / static_cast<double>(slot.samples.size()));
      slot.eligible = slot.launch_ok && slot.occupancy >= 1 &&
                      slot.envelope.ok && slot.timed_nonfinite == 0 &&
                      slot.samples.size() == static_cast<std::size_t>(kMeasured);
      if (slot.eligible && slot.mean_ms < best) {
        best = slot.mean_ms;
        winner = tile;
      }
    }
    results[proj].winner = winner;
    const Candidate& baseline = results[proj].candidates[kBaseline];
    const Candidate& chosen = results[proj].candidates[winner];
    results[proj].win = winner != kBaseline && chosen.eligible &&
                        baseline.eligible && chosen.mean_ms < baseline.mean_ms;

    cudaFree(device_out);
    cudaFree(device_y);
    cudaFree(device_p);
    cudaFree(device_w);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  std::fclose(raw);

  bool any_win = false;
  for (int proj = 0; proj < kProjectionCount; ++proj) {
    any_win = any_win || results[proj].win;
  }

  std::time_t now = std::time(nullptr);
  std::tm utc{};
  gmtime_r(&now, &utc);
  char utc_text[32];
  std::strftime(utc_text, sizeof(utc_text), "%Y-%m-%dT%H:%M:%SZ", &utc);

  std::printf("%s{", kPrefix);
  std::printf("\"measurement_utc\":\"%s\",\"device\":\"%s\","
              "\"compute_capability\":\"%d.%d\",\"any_win\":%s,\"projections\":{",
              utc_text, props.name, props.major, props.minor,
              json_bool(any_win));
  for (int proj = 0; proj < kProjectionCount; ++proj) {
    if (proj != 0) std::printf(",");
    const Projection& shape = kProjections[proj];
    const ProjectionResult& result = results[proj];
    const int installed = result.win ? result.winner : kBaseline;
    std::printf("\"%s\":{\"id\":\"%s\",\"rows\":%zu,\"columns\":%zu,"
                "\"prompt_rows\":4096,\"winner\":\"%s\",\"win\":%s,"
                "\"installed\":\"%s\",\"baseline_id\":\"i128_j128\","
                "\"candidates\":{",
                shape.id, shape.id, shape.rows, shape.columns,
                kTiles[result.winner].id, json_bool(result.win),
                kTiles[installed].id);
    for (int tile = 0; tile < kTileCount; ++tile) {
      if (tile != 0) std::printf(",");
      const Candidate& slot = result.candidates[tile];
      std::printf("\"%s\":{\"id\":\"%s\",\"quality_i\":%u,\"prompt_tile\":%u,"
                  "\"mean_ms\":%.9g,",
                  kTiles[tile].id, kTiles[tile].id, kTiles[tile].quality_i,
                  kTiles[tile].prompt_tile, static_cast<double>(slot.mean_ms));
      print_array(stdout, "samples", slot.samples);
      std::printf(",");
      print_array(stdout, "warmup_ms", slot.warmup);
      std::printf(
          ",\"occupancy\":%d,\"launch_ok\":%s,\"eligible\":%s,"
          "\"timed_nonfinite\":%zu,\"envelope\":{\"max_abs\":%.9g,\"rms\":%.9g,"
          "\"nonfinite\":%zu,\"bad\":%zu,\"ok\":%s}}",
          slot.occupancy, json_bool(slot.launch_ok), json_bool(slot.eligible),
          slot.timed_nonfinite, static_cast<double>(slot.envelope.max_abs),
          static_cast<double>(slot.envelope.rms), slot.envelope.nonfinite,
          slot.envelope.bad, json_bool(slot.envelope.ok));
    }
    std::printf("}}");
  }
  std::printf("}}\n");
  std::printf("status=passed\n");
  return 0;
}
