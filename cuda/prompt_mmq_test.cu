#include "quant_mmv.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

#include "quant.h"

namespace {

constexpr int kWarmups = 3;
constexpr int kSamples = 30;
constexpr int kReplicates = 3;
constexpr int kThreads = 256;
constexpr unsigned int kTiles[] = {1U, 2U, 4U, 8U, 16U, 32U, 64U};
constexpr std::size_t kBuckets[] = {1, 2, 4, 8, 16, 32, 64, 256, 4096};

int fail_cuda(const char* operation, cudaError_t error) {
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
  return 1;
}

void write_u16(std::uint8_t* bytes, std::uint16_t value) {
  bytes[0] = static_cast<std::uint8_t>(value & 0xFFU);
  bytes[1] = static_cast<std::uint8_t>(value >> 8U);
}

std::size_t weight_bytes(qw38::cuda::QuantKind kind, std::size_t rows,
                         std::size_t columns) {
  const std::size_t values =
      kind == qw38::cuda::QuantKind::kQ8_0 ? 32 : 256;
  const std::size_t bytes = kind == qw38::cuda::QuantKind::kQ4K
                                ? 144
                                : kind == qw38::cuda::QuantKind::kQ6K ? 210 : 34;
  return rows * (columns / values) * bytes;
}

const char* kind_name(qw38::cuda::QuantKind kind) {
  if (kind == qw38::cuda::QuantKind::kQ4K) return "q4_k";
  if (kind == qw38::cuda::QuantKind::kQ6K) return "q6_k";
  return "q8_0";
}

void fill_weights(qw38::cuda::QuantKind kind, std::size_t rows,
                  std::size_t columns, std::vector<std::uint8_t>* weights) {
  const std::size_t block_bytes =
      kind == qw38::cuda::QuantKind::kQ4K
          ? 144
          : kind == qw38::cuda::QuantKind::kQ6K ? 210 : 34;
  weights->resize(weight_bytes(kind, rows, columns));
  for (std::size_t index = 0; index < weights->size(); ++index) {
    (*weights)[index] = static_cast<std::uint8_t>((index * 73 + 19) & 0xFFU);
  }
  for (std::size_t offset = 0; offset < weights->size();
       offset += block_bytes) {
    if (kind == qw38::cuda::QuantKind::kQ4K) {
      write_u16(weights->data() + offset, 0x2400U);
      write_u16(weights->data() + offset + 2, 0x1C00U);
    } else if (kind == qw38::cuda::QuantKind::kQ6K) {
      write_u16(weights->data() + offset + 208, 0x1C00U);
    } else {
      write_u16(weights->data() + offset, 0x3400U);
      for (std::size_t lane = 0; lane < 32; ++lane) {
        (*weights)[offset + 2 + lane] = static_cast<std::uint8_t>(
            static_cast<std::int8_t>(
                static_cast<int>((lane * 11 + offset) % 63) - 31));
      }
    }
  }
}

void make_activation(std::size_t columns, std::size_t prompt_row,
                     std::vector<__nv_bfloat16>* activation,
                     std::vector<qw38::cuda::Q8Block>* staged) {
  activation->resize(columns);
  staged->resize(columns / 32);
  for (std::size_t index = 0; index < columns; ++index) {
    const float phase = static_cast<float>(index) * 0.071F +
                        static_cast<float>(prompt_row % 5) * 0.173F;
    const float value =
        std::sin(phase) * 3.0F +
        static_cast<float>(static_cast<int>(index % 11) - 5) * 0.03125F +
        static_cast<float>(static_cast<int>(prompt_row % 5) - 2) * 0.015625F;
    (*activation)[index] = __float2bfloat16_rn(value);
  }
  for (std::size_t block = 0; block < staged->size(); ++block) {
    float maximum = 0.0F;
    for (std::size_t lane = 0; lane < 32; ++lane) {
      maximum = std::max(
          maximum,
          std::fabs(__bfloat162float((*activation)[block * 32 + lane])));
    }
    const float scale = maximum == 0.0F ? 0.0F : maximum / 127.0F;
    (*staged)[block].scale = scale;
    for (std::size_t lane = 0; lane < 32; ++lane) {
      const float value =
          __bfloat162float((*activation)[block * 32 + lane]);
      (*staged)[block].values[lane] =
          scale == 0.0F ? 0
                        : static_cast<std::int8_t>(std::round(value / scale));
    }
  }
}

bool reference_dequant_gemm_mmq(qw38::cuda::QuantKind kind,
                                const std::vector<std::uint8_t>& weights,
                                std::size_t output_rows, std::size_t columns,
                                const std::vector<__nv_bfloat16>& prompt,
                                std::size_t prompt_rows,
                                std::vector<float>* output) {
  const std::size_t block_bytes =
      kind == qw38::cuda::QuantKind::kQ4K
          ? 144
          : kind == qw38::cuda::QuantKind::kQ6K ? 210 : 34;
  const std::size_t block_values =
      kind == qw38::cuda::QuantKind::kQ8_0 ? 32 : 256;
  output->assign(prompt_rows * output_rows, 0.0F);
  std::vector<float> decoded(block_values);
  for (std::size_t out = 0; out < output_rows; ++out) {
    for (std::size_t prompt_row = 0; prompt_row < prompt_rows; ++prompt_row) {
      float sum = 0.0F;
      for (std::size_t block = 0; block < columns / block_values; ++block) {
        const std::uint8_t* packed =
            weights.data() +
            (out * (columns / block_values) + block) * block_bytes;
        const qw38::Status status =
            kind == qw38::cuda::QuantKind::kQ4K
                ? qw38::internal::decode_q4_k(packed, block_bytes,
                                              decoded.data(), decoded.size())
            : kind == qw38::cuda::QuantKind::kQ6K
                ? qw38::internal::decode_q6_k(packed, block_bytes,
                                              decoded.data(), decoded.size())
                : qw38::internal::decode_q8_0(packed, block_bytes,
                                              decoded.data(), decoded.size());
        if (!status.is_ok()) return false;
        for (std::size_t within = 0; within < block_values; ++within) {
          const std::size_t column = block * block_values + within;
          sum += decoded[within] * __bfloat162float(
                     prompt[prompt_row * columns + column]);
        }
      }
      (*output)[prompt_row * output_rows + out] = sum;
    }
  }
  return true;
}

bool ds4_q4k_association_ok(const std::vector<float>& got,
                            const std::vector<float>& ref, std::size_t columns) {
  constexpr float kAbsScale = 0.20F;
  constexpr float kRelTol = 0.05F;
  const float abs_tol = kAbsScale * std::sqrt(static_cast<float>(columns));
  for (std::size_t index = 0; index < got.size(); ++index) {
    if (!std::isfinite(got[index]) || !std::isfinite(ref[index])) return false;
    const float absolute = std::fabs(got[index] - ref[index]);
    const float relative =
        ref[index] != 0.0F ? absolute / std::fabs(ref[index])
                           : (absolute > 0.0F ? INFINITY : 0.0F);
    if (absolute > abs_tol && relative > kRelTol) return false;
  }
  return true;
}

struct LaunchInfo {
  dim3 grid{};
  dim3 block{};
  int nodes = 0;
  int registers = 0;
  int local_bytes = 0;
  int static_shared = 0;
  int active_blocks = 0;
  unsigned grid_x = 0;
  unsigned grid_y = 0;
};

template <typename Fn>
cudaError_t capture_mmq(Fn launch, std::size_t output_rows, LaunchInfo* info) {
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaError_t error =
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (error == cudaSuccess) {
    error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  }
  if (error == cudaSuccess) error = launch(stream);
  if (error == cudaSuccess) error = cudaStreamEndCapture(stream, &graph);
  std::size_t count = 0;
  if (error == cudaSuccess) error = cudaGraphGetNodes(graph, nullptr, &count);
  std::vector<cudaGraphNode_t> nodes(count);
  if (error == cudaSuccess) {
    error = cudaGraphGetNodes(graph, nodes.data(), &count);
  }
  info->nodes = static_cast<int>(count);
  void* function = nullptr;
  for (cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type{};
    cudaKernelNodeParams params{};
    if (error != cudaSuccess) break;
    if (cudaGraphNodeGetType(node, &type) != cudaSuccess ||
        type != cudaGraphNodeTypeKernel ||
        cudaGraphKernelNodeGetParams(node, &params) != cudaSuccess) {
      error = cudaErrorInvalidValue;
      break;
    }
    if (params.gridDim.x == static_cast<unsigned>((output_rows + 7) / 8) ||
        params.gridDim.x ==
            static_cast<unsigned>((output_rows + 127) / 128) ||
        params.gridDim.x ==
            static_cast<unsigned>((output_rows + 63) / 64) ||
        params.gridDim.x ==
            static_cast<unsigned>((output_rows + 31) / 32)) {
      info->grid = params.gridDim;
      info->block = params.blockDim;
      info->grid_x = params.gridDim.x;
      info->grid_y = params.gridDim.y;
      function = params.func;
    }
  }
  if (error == cudaSuccess && function != nullptr) {
    cudaFuncAttributes attr{};
    error = cudaFuncGetAttributes(&attr, function);
    info->registers = attr.numRegs;
    info->local_bytes = static_cast<int>(attr.localSizeBytes);
    info->static_shared = static_cast<int>(attr.sharedSizeBytes);
    const unsigned mma_grid_x =
        static_cast<unsigned>((output_rows + 127) / 128);
    const std::size_t dynamic =
        info->grid_x == mma_grid_x
            ? qw38::cuda::mma_mmq_shared_bytes(
                  qw38::cuda::selected_mma_mmq_prompt_tile())
            : 0;
    const int occupancy_threads =
        info->block.x * info->block.y * info->block.z;
    if (error == cudaSuccess) {
      error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &info->active_blocks, function,
          occupancy_threads == 0 ? kThreads : occupancy_threads, dynamic);
    }
  }
  if (graph != nullptr) cudaGraphDestroy(graph);
  if (stream != nullptr) cudaStreamDestroy(stream);
  return error;
}

float time_launch(int warmups, int samples,
                  cudaError_t (*body)(void*), void* context,
                  cudaError_t* error) {
  for (int warmup = 0; warmup < warmups && *error == cudaSuccess; ++warmup) {
    *error = body(context);
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (*error == cudaSuccess) *error = cudaEventCreate(&start);
  if (*error == cudaSuccess) *error = cudaEventCreate(&stop);
  float total = 0.0F;
  for (int sample = 0; sample < samples && *error == cudaSuccess; ++sample) {
    *error = cudaEventRecord(start);
    if (*error == cudaSuccess) *error = body(context);
    if (*error == cudaSuccess) *error = cudaEventRecord(stop);
    if (*error == cudaSuccess) *error = cudaEventSynchronize(stop);
    float milliseconds = 0.0F;
    if (*error == cudaSuccess) {
      *error = cudaEventElapsedTime(&milliseconds, start, stop);
    }
    if (*error == cudaSuccess) total += milliseconds;
  }
  if (start != nullptr) cudaEventDestroy(start);
  if (stop != nullptr) cudaEventDestroy(stop);
  return samples > 0 ? total / static_cast<float>(samples) : 0.0F;
}

bool outputs_equal(const std::vector<float>& left,
                   const std::vector<float>& right) {
  if (left.size() != right.size()) return false;
  return std::memcmp(left.data(), right.data(),
                     left.size() * sizeof(float)) == 0;
}

bool q8_blocks_equal(const std::vector<qw38::cuda::Q8Block>& left,
                     const std::vector<qw38::cuda::Q8Block>& right) {
  if (left.size() != right.size()) return false;
  for (std::size_t index = 0; index < left.size(); ++index) {
    if (left[index].scale != right[index].scale) return false;
    if (std::memcmp(left[index].values, right[index].values, 32) != 0) {
      return false;
    }
  }
  return true;
}

bool run_q8_exact() {
  constexpr std::size_t kPromptRows[] = {1, 3, 5, 8, 9, 64, 65};
  struct Shape {
    std::size_t output_rows;
    std::size_t columns;
  };
  constexpr Shape kShapes[] = {{17, 256}, {1024, 5120}};
  for (Shape shape : kShapes) {
    std::vector<std::uint8_t> weights;
    fill_weights(qw38::cuda::QuantKind::kQ8_0, shape.output_rows,
                 shape.columns, &weights);
    std::uint8_t* device_weights = nullptr;
    if (cudaMalloc(&device_weights, weights.size()) != cudaSuccess) return false;
    if (cudaMemcpy(device_weights, weights.data(), weights.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      cudaFree(device_weights);
      return false;
    }
    for (std::size_t prompt_rows : kPromptRows) {
      std::vector<__nv_bfloat16> prompt(prompt_rows * shape.columns);
      for (std::size_t row = 0; row < prompt_rows; ++row) {
        std::vector<__nv_bfloat16> activation;
        std::vector<qw38::cuda::Q8Block> staged;
        make_activation(shape.columns, row, &activation, &staged);
        std::copy(activation.begin(), activation.end(),
                  prompt.begin() + row * shape.columns);
      }
      __nv_bfloat16* device_prompt = nullptr;
      float* device_prod = nullptr;
      float* device_ref = nullptr;
      const std::size_t values = prompt_rows * shape.output_rows;
      if (cudaMalloc(&device_prompt, prompt.size() * sizeof(prompt[0])) !=
              cudaSuccess ||
          cudaMalloc(&device_prod, values * sizeof(float)) != cudaSuccess ||
          cudaMalloc(&device_ref, values * sizeof(float)) != cudaSuccess) {
        cudaFree(device_ref);
        cudaFree(device_prod);
        cudaFree(device_prompt);
        cudaFree(device_weights);
        return false;
      }
      cudaMemcpy(device_prompt, prompt.data(),
                 prompt.size() * sizeof(prompt[0]), cudaMemcpyHostToDevice);
      const unsigned tile = qw38::cuda::selected_mmq_prompt_tile(
          qw38::cuda::QuantKind::kQ8_0, prompt_rows);
      const cudaError_t produced = qw38::cuda::launch_q8_mmq_bf16_variant(
          device_weights, shape.output_rows, shape.columns, device_prompt,
          prompt_rows, device_prod, tile, nullptr);
      const cudaError_t referenced = qw38::cuda::launch_q8_mmq_bf16_reference(
          device_weights, shape.output_rows, shape.columns, device_prompt,
          prompt_rows, device_ref, nullptr);
      std::vector<float> actual(values);
      std::vector<float> expected(values);
      const bool ok =
          produced == cudaSuccess && referenced == cudaSuccess &&
          cudaDeviceSynchronize() == cudaSuccess &&
          cudaMemcpy(actual.data(), device_prod, values * sizeof(float),
                     cudaMemcpyDeviceToHost) == cudaSuccess &&
          cudaMemcpy(expected.data(), device_ref, values * sizeof(float),
                     cudaMemcpyDeviceToHost) == cudaSuccess &&
          outputs_equal(actual, expected);
      cudaFree(device_ref);
      cudaFree(device_prod);
      cudaFree(device_prompt);
      if (!ok) {
        cudaFree(device_weights);
        return false;
      }
    }
    cudaFree(device_weights);
  }
  return true;
}

bool run_quant_tiles(qw38::cuda::QuantKind kind, std::size_t output_rows,
                     std::size_t columns, std::size_t prompt_rows,
                     bool* envelope_ok, bool* staging_ok) {
  std::vector<std::uint8_t> weights;
  fill_weights(kind, output_rows, columns, &weights);
  std::vector<__nv_bfloat16> prompt(prompt_rows * columns);
  std::vector<qw38::cuda::Q8Block> staged(prompt_rows * (columns / 32));
  for (std::size_t row = 0; row < prompt_rows; ++row) {
    std::vector<__nv_bfloat16> activation;
    std::vector<qw38::cuda::Q8Block> row_staged;
    make_activation(columns, row, &activation, &row_staged);
    std::copy(activation.begin(), activation.end(),
              prompt.begin() + row * columns);
    std::copy(row_staged.begin(), row_staged.end(),
              staged.begin() + row * (columns / 32));
  }
  std::vector<float> expected;
  if (!reference_dequant_gemm_mmq(kind, weights, output_rows, columns, prompt,
                                  prompt_rows, &expected)) {
    return false;
  }
  std::uint8_t* device_weights = nullptr;
  __nv_bfloat16* device_prompt = nullptr;
  qw38::cuda::Q8Block* device_staged = nullptr;
  float* device_output = nullptr;
  if (cudaMalloc(&device_weights, weights.size()) != cudaSuccess ||
      cudaMalloc(&device_prompt, prompt.size() * sizeof(prompt[0])) !=
          cudaSuccess ||
      cudaMalloc(&device_staged, staged.size() * sizeof(staged[0])) !=
          cudaSuccess ||
      cudaMalloc(&device_output, expected.size() * sizeof(float)) !=
          cudaSuccess) {
    cudaFree(device_output);
    cudaFree(device_staged);
    cudaFree(device_prompt);
    cudaFree(device_weights);
    return false;
  }
  cudaMemcpy(device_weights, weights.data(), weights.size(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(device_prompt, prompt.data(), prompt.size() * sizeof(prompt[0]),
             cudaMemcpyHostToDevice);
  const unsigned selected =
      qw38::cuda::selected_mmq_prompt_tile(kind, prompt_rows);
  const unsigned tiles[3] = {1U, 8U, selected};
  std::vector<float> first;
  bool identical = true;
  *envelope_ok = true;
  *staging_ok = true;
  for (unsigned tile : tiles) {
    cudaMemset(device_staged, 0, staged.size() * sizeof(staged[0]));
    if (qw38::cuda::launch_quant_mmq_variant(
            kind, device_weights, output_rows, columns, device_prompt,
            prompt_rows, device_staged, device_output, tile, nullptr) !=
            cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
      identical = false;
      break;
    }
    std::vector<float> actual(expected.size());
    std::vector<qw38::cuda::Q8Block> actual_staged(staged.size());
    cudaMemcpy(actual.data(), device_output, actual.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(actual_staged.data(), device_staged,
               actual_staged.size() * sizeof(actual_staged[0]),
               cudaMemcpyDeviceToHost);
    if (first.empty()) {
      first = actual;
    } else if (!outputs_equal(first, actual)) {
      identical = false;
    }
    if (!q8_blocks_equal(actual_staged, staged)) *staging_ok = false;
    if (!ds4_q4k_association_ok(actual, expected, columns)) {
      *envelope_ok = false;
    }
  }
  cudaFree(device_output);
  cudaFree(device_staged);
  cudaFree(device_prompt);
  cudaFree(device_weights);
  return identical;
}

struct TimingCtx {
  qw38::cuda::QuantKind kind = qw38::cuda::QuantKind::kQ8_0;
  const std::uint8_t* weights = nullptr;
  std::size_t output_rows = 0;
  std::size_t columns = 0;
  const __nv_bfloat16* prompt = nullptr;
  std::size_t prompt_rows = 0;
  qw38::cuda::Q8Block* staged = nullptr;
  float* output = nullptr;
  unsigned tile = 1;
  bool q8_bf16 = false;
  bool reference = false;
};

cudaError_t timing_body(void* context) {
  auto* ctx = static_cast<TimingCtx*>(context);
  if (ctx->q8_bf16) {
    if (ctx->reference) {
      return qw38::cuda::launch_q8_mmq_bf16_reference(
          ctx->weights, ctx->output_rows, ctx->columns, ctx->prompt,
          ctx->prompt_rows, ctx->output, nullptr);
    }
    if (ctx->tile == 0) {
      return qw38::cuda::launch_q8_mmq_bf16(
          ctx->weights, ctx->output_rows, ctx->columns, ctx->prompt,
          ctx->prompt_rows, ctx->staged, ctx->output, nullptr);
    }
    return qw38::cuda::launch_q8_mmq_bf16_variant(
        ctx->weights, ctx->output_rows, ctx->columns, ctx->prompt,
        ctx->prompt_rows, ctx->output, ctx->tile, nullptr);
  }
  if (ctx->tile == 0) {
    return qw38::cuda::launch_quant_mmq(
        ctx->kind, ctx->weights, ctx->output_rows, ctx->columns, ctx->prompt,
        ctx->prompt_rows, ctx->staged, ctx->output, nullptr);
  }
  return qw38::cuda::launch_quant_mmq_variant(
      ctx->kind, ctx->weights, ctx->output_rows, ctx->columns, ctx->prompt,
      ctx->prompt_rows, ctx->staged, ctx->output, ctx->tile, nullptr);
}

int occupancy_for_tile(bool q8_bf16, qw38::cuda::QuantKind kind,
                       unsigned tile) {
  const std::size_t columns = 256;
  const std::size_t output_rows = 8;
  const std::size_t prompt_rows = tile;
  std::uint8_t* weights = nullptr;
  __nv_bfloat16* prompt = nullptr;
  qw38::cuda::Q8Block* staged = nullptr;
  float* output = nullptr;
  const auto kind_for_bytes =
      q8_bf16 ? qw38::cuda::QuantKind::kQ8_0 : kind;
  cudaMalloc(&weights, weight_bytes(kind_for_bytes, output_rows, columns));
  cudaMalloc(&prompt, prompt_rows * columns * sizeof(__nv_bfloat16));
  cudaMalloc(&output, prompt_rows * output_rows * sizeof(float));
  if (!q8_bf16) {
    cudaMalloc(&staged,
               qw38::cuda::q8_prompt_workspace_bytes(prompt_rows, columns));
  }
  LaunchInfo info{};
  cudaError_t error = capture_mmq(
      [&](cudaStream_t stream) {
        if (q8_bf16) {
          return qw38::cuda::launch_q8_mmq_bf16_variant(
              weights, output_rows, columns, prompt, prompt_rows, output, tile,
              stream);
        }
        return qw38::cuda::launch_quant_mmq_variant(
            kind, weights, output_rows, columns, prompt, prompt_rows, staged,
            output, tile, stream);
      },
      output_rows, &info);
  cudaFree(output);
  cudaFree(staged);
  cudaFree(prompt);
  cudaFree(weights);
  return error == cudaSuccess ? info.active_blocks : 0;
}

int run_sweep() {
  struct Shape {
    qw38::cuda::QuantKind kind;
    std::size_t output_rows;
    std::size_t columns;
    bool q8_bf16;
  };
  constexpr Shape kShapes[] = {
      {qw38::cuda::QuantKind::kQ8_0, 12288, 5120, true},
      {qw38::cuda::QuantKind::kQ4K, 17408, 5120, false},
      {qw38::cuda::QuantKind::kQ4K, 5120, 17408, false},
      {qw38::cuda::QuantKind::kQ6K, 5120, 6144, false},
  };
  int occupancy[4][7]{};
  for (int shape = 0; shape < 4; ++shape) {
    for (int tile = 0; tile < 7; ++tile) {
      occupancy[shape][tile] = occupancy_for_tile(
          kShapes[shape].q8_bf16, kShapes[shape].kind, kTiles[tile]);
    }
  }
  for (int replicate = 1; replicate <= kReplicates; ++replicate) {
    for (int shape = 0; shape < 4; ++shape) {
      const Shape& spec = kShapes[shape];
      std::uint8_t* weights = nullptr;
      __nv_bfloat16* prompt = nullptr;
      qw38::cuda::Q8Block* staged = nullptr;
      float* output = nullptr;
      const std::size_t max_prompt = 4096;
      cudaError_t error =
          cudaMalloc(&weights, weight_bytes(spec.kind, spec.output_rows,
                                            spec.columns));
      if (error == cudaSuccess) {
        error = cudaMalloc(&prompt, max_prompt * spec.columns *
                                        sizeof(__nv_bfloat16));
      }
      if (error == cudaSuccess && !spec.q8_bf16) {
        error = cudaMalloc(
            &staged, qw38::cuda::q8_prompt_workspace_bytes(max_prompt,
                                                           spec.columns));
      }
      if (error == cudaSuccess) {
        error = cudaMalloc(&output,
                           max_prompt * spec.output_rows * sizeof(float));
      }
      if (error == cudaSuccess) {
        error = cudaMemset(weights, 0,
                           weight_bytes(spec.kind, spec.output_rows,
                                        spec.columns));
      }
      if (error != cudaSuccess) {
        cudaFree(output);
        cudaFree(staged);
        cudaFree(prompt);
        cudaFree(weights);
        return fail_cuda("sweep allocation", error);
      }
      for (std::size_t prompt_rows : kBuckets) {
        error = cudaMemset(prompt, 0,
                           prompt_rows * spec.columns * sizeof(__nv_bfloat16));
        if (error == cudaSuccess) {
          error = cudaMemset(output, 0x7F,
                             prompt_rows * spec.output_rows * sizeof(float));
        }
        for (int tile_index = 0; tile_index < 7; ++tile_index) {
          const unsigned tile = kTiles[tile_index];
          const int active = occupancy[shape][tile_index];
          TimingCtx ctx{};
          ctx.kind = spec.kind;
          ctx.weights = weights;
          ctx.output_rows = spec.output_rows;
          ctx.columns = spec.columns;
          ctx.prompt = prompt;
          ctx.prompt_rows = prompt_rows;
          ctx.staged = staged;
          ctx.output = output;
          ctx.tile = tile;
          ctx.q8_bf16 = spec.q8_bf16;
          cudaError_t launch_error = cudaSuccess;
          for (int warmup = 0;
               warmup < kWarmups && launch_error == cudaSuccess; ++warmup) {
            launch_error = timing_body(&ctx);
          }
          cudaEvent_t start = nullptr;
          cudaEvent_t stop = nullptr;
          if (launch_error == cudaSuccess) {
            launch_error = cudaEventCreate(&start);
          }
          if (launch_error == cudaSuccess) launch_error = cudaEventCreate(&stop);
          for (int sample = 0; sample < kSamples; ++sample) {
            float milliseconds = 0.0F;
            cudaError_t sample_error = launch_error;
            if (sample_error == cudaSuccess) {
              sample_error = cudaEventRecord(start);
            }
            if (sample_error == cudaSuccess) sample_error = timing_body(&ctx);
            if (sample_error == cudaSuccess) {
              sample_error = cudaEventRecord(stop);
            }
            if (sample_error == cudaSuccess) {
              sample_error = cudaEventSynchronize(stop);
            }
            if (sample_error == cudaSuccess) {
              sample_error = cudaEventElapsedTime(&milliseconds, start, stop);
            }
            const int launched = sample_error == cudaSuccess ? 1 : 0;
            std::printf(
                "tune=mmq kind=%s prompt_rows=%zu output_rows=%zu columns=%zu "
                "tile=%u replicate=%d sample=%d milliseconds=%.9g occupancy=%d "
                "launched=%d\n",
                kind_name(spec.kind), prompt_rows, spec.output_rows,
                spec.columns, tile, replicate, sample,
                launched ? milliseconds : -1.0, active, launched);
            if (sample_error != cudaSuccess) launch_error = sample_error;
          }
          if (start != nullptr) cudaEventDestroy(start);
          if (stop != nullptr) cudaEventDestroy(stop);
          int nonzero = 1;
          if (launch_error == cudaSuccess &&
              cudaDeviceSynchronize() == cudaSuccess) {
            float edges[2] = {1.0F, 1.0F};
            if (cudaMemcpy(&edges[0], output, sizeof(float),
                           cudaMemcpyDeviceToHost) == cudaSuccess &&
                cudaMemcpy(&edges[1],
                           output + prompt_rows * spec.output_rows - 1,
                           sizeof(float), cudaMemcpyDeviceToHost) ==
                    cudaSuccess) {
              nonzero = (edges[0] != 0.0F || edges[1] != 0.0F) ? 1 : 0;
            }
          }
          const int eligible =
              launch_error == cudaSuccess && active >= 1 && nonzero == 0 &&
              (prompt_rows < 2 || tile >= 2);
          std::printf(
              "tune=summary kind=%s prompt_rows=%zu output_rows=%zu "
              "columns=%zu tile=%u replicate=%d occupancy=%d nonzero=%d "
              "eligible=%d\n",
              kind_name(spec.kind), prompt_rows, spec.output_rows,
              spec.columns, tile, replicate, active, nonzero, eligible);
        }
      }
      cudaFree(output);
      cudaFree(staged);
      cudaFree(prompt);
      cudaFree(weights);
    }
  }
  std::printf("status=passed\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::strcmp(argv[1], "--sweep") == 0) {
    return run_sweep();
  }

  const bool q8_exact = run_q8_exact();
  bool q4_identical = true;
  bool q6_identical = true;
  bool envelope_ok = true;
  bool staging_ok = true;
  struct QuantCase {
    qw38::cuda::QuantKind kind;
    std::size_t output_rows;
    std::size_t columns;
    std::size_t prompt_rows;
  };
  constexpr QuantCase kQuantCases[] = {
      {qw38::cuda::QuantKind::kQ4K, 17, 256, 3},
      {qw38::cuda::QuantKind::kQ4K, 257, 512, 5},
      {qw38::cuda::QuantKind::kQ4K, 257, 512, 64},
      {qw38::cuda::QuantKind::kQ6K, 17, 256, 1},
      {qw38::cuda::QuantKind::kQ6K, 257, 512, 9},
      {qw38::cuda::QuantKind::kQ6K, 257, 512, 64},
  };
  for (const QuantCase& test : kQuantCases) {
    bool case_envelope = true;
    bool case_staging = true;
    const bool identical = run_quant_tiles(
        test.kind, test.output_rows, test.columns, test.prompt_rows,
        &case_envelope, &case_staging);
    if (test.kind == qw38::cuda::QuantKind::kQ4K) {
      q4_identical = q4_identical && identical;
    } else {
      q6_identical = q6_identical && identical;
    }
    envelope_ok = envelope_ok && case_envelope;
    staging_ok = staging_ok && case_staging;
  }

  const bool invalid =
      qw38::cuda::launch_q8_mmq_bf16(nullptr, 17, 256, nullptr, 1, nullptr,
                                     nullptr, nullptr) == cudaErrorInvalidValue &&
      qw38::cuda::launch_q8_mmq_bf16_variant(
          nullptr, 17, 256, nullptr, 1, nullptr, 3, nullptr) ==
          cudaErrorInvalidValue &&
      qw38::cuda::launch_q8_mmq_bf16_reference(nullptr, 0, 255, nullptr, 0,
                                               nullptr, nullptr) ==
          cudaErrorInvalidValue &&
      qw38::cuda::launch_quant_mmq_variant(
          qw38::cuda::QuantKind::kQ4K, nullptr, 17, 256, nullptr, 1, nullptr,
          nullptr, 3, nullptr) == cudaErrorInvalidValue;

  std::uint8_t* q8_weights = nullptr;
  __nv_bfloat16* q8_prompt = nullptr;
  qw38::cuda::Q8Block* q8_staged = nullptr;
  float* q8_output = nullptr;
  std::uint8_t* q4_weights = nullptr;
  __nv_bfloat16* q4_prompt = nullptr;
  qw38::cuda::Q8Block* q4_staged = nullptr;
  float* q4_output = nullptr;
  std::uint8_t* q4_down_weights = nullptr;
  __nv_bfloat16* q4_down_prompt = nullptr;
  qw38::cuda::Q8Block* q4_down_staged = nullptr;
  float* q4_down_output = nullptr;
  std::uint8_t* q6_weights = nullptr;
  __nv_bfloat16* q6_prompt = nullptr;
  qw38::cuda::Q8Block* q6_staged = nullptr;
  float* q6_output = nullptr;
  const std::size_t q8_rows = 12288;
  const std::size_t q8_cols = 5120;
  const std::size_t q4_rows = 17408;
  const std::size_t q4_cols = 5120;
  const std::size_t q6_rows = 5120;
  const std::size_t q6_cols = 6144;
  cudaMalloc(&q8_weights,
             weight_bytes(qw38::cuda::QuantKind::kQ8_0, q8_rows, q8_cols));
  cudaMalloc(&q8_prompt, 4096 * q8_cols * sizeof(__nv_bfloat16));
  cudaMalloc(&q8_staged,
             qw38::cuda::q8_prompt_workspace_bytes(4096, q8_cols));
  cudaMalloc(&q8_output, 4096 * q8_rows * sizeof(float));
  cudaMalloc(&q4_weights,
             weight_bytes(qw38::cuda::QuantKind::kQ4K, q4_rows, q4_cols));
  cudaMalloc(&q4_prompt, 4096 * q4_cols * sizeof(__nv_bfloat16));
  cudaMalloc(&q4_staged,
             qw38::cuda::q8_prompt_workspace_bytes(4096, q4_cols));
  cudaMalloc(&q4_output, 4096 * q4_rows * sizeof(float));
  cudaMalloc(&q4_down_weights,
             weight_bytes(qw38::cuda::QuantKind::kQ4K, q4_cols, q4_rows));
  cudaMalloc(&q4_down_prompt, 4096 * q4_rows * sizeof(__nv_bfloat16));
  cudaMalloc(&q4_down_staged,
             qw38::cuda::q8_prompt_workspace_bytes(4096, q4_rows));
  cudaMalloc(&q4_down_output, 4096 * q4_cols * sizeof(float));
  cudaMalloc(&q6_weights,
             weight_bytes(qw38::cuda::QuantKind::kQ6K, q6_rows, q6_cols));
  cudaMalloc(&q6_prompt, 4096 * q6_cols * sizeof(__nv_bfloat16));
  cudaMalloc(&q6_staged,
             qw38::cuda::q8_prompt_workspace_bytes(4096, q6_cols));
  cudaMalloc(&q6_output, 4096 * q6_rows * sizeof(float));
  cudaMemset(q8_weights, 0,
             weight_bytes(qw38::cuda::QuantKind::kQ8_0, q8_rows, q8_cols));
  cudaMemset(q8_prompt, 0, 4096 * q8_cols * sizeof(__nv_bfloat16));
  cudaMemset(q4_weights, 0,
             weight_bytes(qw38::cuda::QuantKind::kQ4K, q4_rows, q4_cols));
  cudaMemset(q4_prompt, 0, 4096 * q4_cols * sizeof(__nv_bfloat16));
  cudaMemset(q4_down_weights, 0,
             weight_bytes(qw38::cuda::QuantKind::kQ4K, q4_cols, q4_rows));
  cudaMemset(q4_down_prompt, 0, 4096 * q4_rows * sizeof(__nv_bfloat16));
  cudaMemset(q6_weights, 0,
             weight_bytes(qw38::cuda::QuantKind::kQ6K, q6_rows, q6_cols));
  cudaMemset(q6_prompt, 0, 4096 * q6_cols * sizeof(__nv_bfloat16));

  LaunchInfo q8_prod_64{};
  LaunchInfo q8_prod_4096{};
  LaunchInfo q8_ref_64{};
  LaunchInfo q8_ref_4096{};
  LaunchInfo q4_64{};
  LaunchInfo q4_4096{};
  LaunchInfo q6_64{};
  LaunchInfo q6_4096{};
  const cudaError_t cap_q8_64 = capture_mmq(
      [&](cudaStream_t stream) {
        return qw38::cuda::launch_q8_mmq_bf16_variant(
            q8_weights, q8_rows, q8_cols, q8_prompt, 64, q8_output,
            qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ8_0,
                                                 64),
            stream);
      },
      q8_rows, &q8_prod_64);
  const cudaError_t cap_q8_4096 = capture_mmq(
      [&](cudaStream_t stream) {
        return qw38::cuda::launch_q8_mmq_bf16_variant(
            q8_weights, q8_rows, q8_cols, q8_prompt, 4096, q8_output,
            qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ8_0,
                                                 4096),
            stream);
      },
      q8_rows, &q8_prod_4096);
  const cudaError_t cap_q8_ref_64 = capture_mmq(
      [&](cudaStream_t stream) {
        return qw38::cuda::launch_q8_mmq_bf16_reference(
            q8_weights, q8_rows, q8_cols, q8_prompt, 64, q8_output, stream);
      },
      q8_rows, &q8_ref_64);
  const cudaError_t cap_q8_ref_4096 = capture_mmq(
      [&](cudaStream_t stream) {
        return qw38::cuda::launch_q8_mmq_bf16_reference(
            q8_weights, q8_rows, q8_cols, q8_prompt, 4096, q8_output, stream);
      },
      q8_rows, &q8_ref_4096);
  const cudaError_t cap_q4_64 = capture_mmq(
      [&](cudaStream_t stream) {
        return qw38::cuda::launch_quant_mmq(
            qw38::cuda::QuantKind::kQ4K, q4_weights, q4_rows, q4_cols,
            q4_prompt, 64, q4_staged, q4_output, stream);
      },
      q4_rows, &q4_64);
  const cudaError_t cap_q4_4096 = capture_mmq(
      [&](cudaStream_t stream) {
        return qw38::cuda::launch_quant_mmq(
            qw38::cuda::QuantKind::kQ4K, q4_weights, q4_rows, q4_cols,
            q4_prompt, 4096, q4_staged, q4_output, stream);
      },
      q4_rows, &q4_4096);
  const cudaError_t cap_q6_64 = capture_mmq(
      [&](cudaStream_t stream) {
        return qw38::cuda::launch_quant_mmq(
            qw38::cuda::QuantKind::kQ6K, q6_weights, q6_rows, q6_cols,
            q6_prompt, 64, q6_staged, q6_output, stream);
      },
      q6_rows, &q6_64);
  const cudaError_t cap_q6_4096 = capture_mmq(
      [&](cudaStream_t stream) {
        return qw38::cuda::launch_quant_mmq(
            qw38::cuda::QuantKind::kQ6K, q6_weights, q6_rows, q6_cols,
            q6_prompt, 4096, q6_staged, q6_output, stream);
      },
      q6_rows, &q6_4096);

  const unsigned q8_tile_64 =
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ8_0, 64);
  const unsigned q8_tile_4096 =
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ8_0, 4096);
  const unsigned q4_tile_64 =
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ4K, 64);
  const unsigned q4_tile_4096 =
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ4K, 4096);
  const unsigned q6_tile_64 =
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ6K, 64);
  const unsigned q6_tile_4096 =
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ6K, 4096);
  const bool q8_grid =
      cap_q8_64 == cudaSuccess && cap_q8_4096 == cudaSuccess &&
      q8_prod_64.nodes == 1 && q8_prod_4096.nodes == 1 &&
      q8_prod_64.grid_x == (q8_rows + 7) / 8 &&
      q8_prod_4096.grid_x == (q8_rows + 7) / 8 &&
      q8_prod_64.grid_y == (64 + q8_tile_64 - 1) / q8_tile_64 &&
      q8_prod_4096.grid_y == (4096 + q8_tile_4096 - 1) / q8_tile_4096 &&
      q8_tile_64 >= 2 && q8_tile_4096 >= 2;
  const bool q8_ref_grid =
      cap_q8_ref_64 == cudaSuccess && cap_q8_ref_4096 == cudaSuccess &&
      q8_ref_64.nodes == 1 && q8_ref_4096.nodes == 1 &&
      q8_ref_64.grid_y == 64 && q8_ref_4096.grid_y == 4096 &&
      q8_ref_64.grid_x == (q8_rows + 7) / 8;
  const unsigned mma_tile = qw38::cuda::selected_mma_mmq_prompt_tile();
  const unsigned q8_quality_tile =
      qw38::cuda::selected_q8_quality_mmq_prompt_tile();
  LaunchInfo q8_quality_64{};
  LaunchInfo q8_quality_4096{};
  const cudaError_t cap_q8_quality_64 = capture_mmq(
      [&](cudaStream_t stream) {
        return qw38::cuda::launch_q8_mmq_bf16(
            q8_weights, q8_rows, q8_cols, q8_prompt, 64, q8_staged, q8_output,
            stream);
      },
      q8_rows, &q8_quality_64);
  const cudaError_t cap_q8_quality_4096 = capture_mmq(
      [&](cudaStream_t stream) {
        return qw38::cuda::launch_q8_mmq_bf16(q8_weights, q8_rows, q8_cols,
                                              q8_prompt, 4096, q8_staged,
                                              q8_output, stream);
      },
      q8_rows, &q8_quality_4096);
  const bool q8_quality_grid =
      cap_q8_quality_64 == cudaSuccess && cap_q8_quality_4096 == cudaSuccess &&
      q8_quality_64.nodes == 2 && q8_quality_4096.nodes == 2 &&
      q8_quality_64.block.x == 32 && q8_quality_64.block.y == 8 &&
      q8_quality_4096.block.x == 32 && q8_quality_4096.block.y == 8 &&
      q8_quality_tile == 128 &&
      q8_quality_64.grid_x == (q8_rows + 127) / 128 &&
      q8_quality_4096.grid_x == (q8_rows + 127) / 128 &&
      q8_quality_64.grid_y == (64 + q8_quality_tile - 1) / q8_quality_tile &&
      q8_quality_4096.grid_y ==
          (4096 + q8_quality_tile - 1) / q8_quality_tile &&
      qw38::cuda::q8_quality_mmq_occupancy(q8_quality_tile) >= 1;
  std::printf("opt022_q8_quality_launch prompt_rows=64 kernel_nodes=%d "
              "block=[%u,%u,1] grid=[%u,%u,1] occupancy=%d\n",
              q8_quality_64.nodes, q8_quality_64.block.x,
              q8_quality_64.block.y, q8_quality_64.grid_x, q8_quality_64.grid_y,
              qw38::cuda::q8_quality_mmq_occupancy(q8_quality_tile));
  std::printf("opt022_q8_quality_launch prompt_rows=4096 kernel_nodes=%d "
              "block=[%u,%u,1] grid=[%u,%u,1]\n",
              q8_quality_4096.nodes, q8_quality_4096.block.x,
              q8_quality_4096.block.y, q8_quality_4096.grid_x,
              q8_quality_4096.grid_y);
  const char* skinny_path = qw38::cuda::selected_skinny_mixer_path();
  const cudaError_t skinny_quant = qw38::cuda::launch_quantize_mmq_q8_1(
      qw38::cuda::QuantKind::kQ8_0, q8_prompt, 64, q8_cols, q8_staged, nullptr);
  LaunchInfo skinny_mma{};
  cudaError_t cap_skinny = cudaErrorInvalidValue;
  if (skinny_quant == cudaSuccess) {
    cap_skinny = capture_mmq(
        [&](cudaStream_t stream) {
          return qw38::cuda::launch_q8_mmq_quality_mma(
              q8_weights, 48, q8_cols, q8_staged, 64, q8_output, stream);
        },
        48, &skinny_mma);
  }
  LaunchInfo skinny_mmv{};
  const cudaError_t cap_skinny_mmv = capture_mmq(
      [&](cudaStream_t stream) {
        return qw38::cuda::launch_q8_mmq_bf16_variant(
            q8_weights, 48, q8_cols, q8_prompt, 64, q8_output, 1, stream);
      },
      48, &skinny_mmv);
  unsigned expected_skinny_grid_x = (48U + 127U) / 128U;
  if (std::strcmp(skinny_path, "mma_i32_j128") == 0) {
    expected_skinny_grid_x = (48U + 31U) / 32U;
  } else if (std::strcmp(skinny_path, "mma_i64_j128") == 0) {
    expected_skinny_grid_x = (48U + 63U) / 64U;
  } else if (std::strcmp(skinny_path, "mmv_tiled_j1") == 0) {
    expected_skinny_grid_x = (48U + 7U) / 8U;
  }
  const bool skinny_mmv_path = std::strcmp(skinny_path, "mmv_tiled_j1") == 0;
  const bool skinny_launch_ok =
      qw38::cuda::skinny_mixer_q8_output_rows(48) &&
      (skinny_mmv_path
           ? (cap_skinny_mmv == cudaSuccess && skinny_mmv.nodes == 1 &&
              skinny_mmv.grid_x == expected_skinny_grid_x)
           : (cap_skinny == cudaSuccess && skinny_mma.nodes == 1 &&
              skinny_mma.grid_x == expected_skinny_grid_x &&
              skinny_mma.block.x == 32 && skinny_mma.block.y == 8));
  std::printf("opt023_skinny_launch path=%s kernel_nodes=%d grid=[%u,%u,1] "
              "block=[%u,%u,1] ok=%s\n",
              skinny_path,
              skinny_mmv_path ? skinny_mmv.nodes : skinny_mma.nodes,
              skinny_mmv_path ? skinny_mmv.grid_x : skinny_mma.grid_x,
              skinny_mmv_path ? skinny_mmv.grid_y : skinny_mma.grid_y,
              skinny_mmv_path ? skinny_mmv.block.x : skinny_mma.block.x,
              skinny_mmv_path ? skinny_mmv.block.y : skinny_mma.block.y,
              skinny_launch_ok ? "true" : "false");
  const bool q4_grid =
      cap_q4_64 == cudaSuccess && cap_q4_4096 == cudaSuccess &&
      q4_64.nodes == 2 && q4_4096.nodes == 2 &&
      q4_64.block.x == 32 && q4_64.block.y == 8 &&
      q4_4096.block.x == 32 && q4_4096.block.y == 8 &&
      q4_64.block.x * q4_64.block.y == 256 &&
      q4_64.grid_x == (q4_rows + 127) / 128 &&
      q4_4096.grid_x == (q4_rows + 127) / 128 &&
      q4_64.grid_y == (64 + mma_tile - 1) / mma_tile &&
      q4_4096.grid_y == (4096 + mma_tile - 1) / mma_tile;
  const bool q6_grid =
      cap_q6_64 == cudaSuccess && cap_q6_4096 == cudaSuccess &&
      q6_64.nodes == 2 && q6_4096.nodes == 2 &&
      q6_64.block.x == 32 && q6_64.block.y == 8 &&
      q6_4096.block.x == 32 && q6_4096.block.y == 8 &&
      q6_64.grid_x == (q6_rows + 127) / 128 &&
      q6_4096.grid_x == (q6_rows + 127) / 128 &&
      q6_64.grid_y == (64 + mma_tile - 1) / mma_tile &&
      q6_4096.grid_y == (4096 + mma_tile - 1) / mma_tile;
  const bool occupancy_ok =
      q8_prod_64.active_blocks >= 1 && q8_prod_64.registers > 0 &&
      q8_prod_64.local_bytes <= 1024 && q4_64.active_blocks >= 1 &&
      q4_64.registers > 0 && q4_64.local_bytes <= 1024 &&
      q6_64.active_blocks >= 1 && q6_64.registers > 0 &&
      q6_64.local_bytes <= 1024 &&
      qw38::cuda::mma_mmq_occupancy(qw38::cuda::QuantKind::kQ4K, mma_tile) >=
          1 &&
      qw38::cuda::mma_mmq_occupancy(qw38::cuda::QuantKind::kQ6K, mma_tile) >= 1;

  cudaError_t timing_error = cudaSuccess;
  TimingCtx q8_prod{};
  q8_prod.weights = q8_weights;
  q8_prod.output_rows = q8_rows;
  q8_prod.columns = q8_cols;
  q8_prod.prompt = q8_prompt;
  q8_prod.output = q8_output;
  q8_prod.staged = q8_staged;
  q8_prod.q8_bf16 = true;
  q8_prod.tile = 0;
  q8_prod.prompt_rows = 64;
  TimingCtx q8_ref = q8_prod;
  q8_ref.reference = true;
  const float q8_64_prod =
      time_launch(kWarmups, kSamples, timing_body, &q8_prod, &timing_error);
  const float q8_64_ref =
      time_launch(kWarmups, kSamples, timing_body, &q8_ref, &timing_error);
  q8_prod.prompt_rows = 256;
  q8_ref.prompt_rows = 256;
  const float q8_256_prod =
      time_launch(kWarmups, kSamples, timing_body, &q8_prod, &timing_error);
  const float q8_256_ref =
      time_launch(kWarmups, kSamples, timing_body, &q8_ref, &timing_error);
  TimingCtx q4_sel{};
  q4_sel.kind = qw38::cuda::QuantKind::kQ4K;
  q4_sel.weights = q4_weights;
  q4_sel.output_rows = q4_rows;
  q4_sel.columns = q4_cols;
  q4_sel.prompt = q4_prompt;
  q4_sel.prompt_rows = 4096;
  q4_sel.staged = q4_staged;
  q4_sel.output = q4_output;
  q4_sel.tile = 0;
  TimingCtx q4_tile8 = q4_sel;
  q4_tile8.tile = 8;
  const float q4_sel_ms =
      time_launch(kWarmups, kSamples, timing_body, &q4_sel, &timing_error);
  const float q4_tile8_ms =
      time_launch(kWarmups, kSamples, timing_body, &q4_tile8, &timing_error);
  TimingCtx q4_down_sel = q4_sel;
  q4_down_sel.weights = q4_down_weights;
  q4_down_sel.output_rows = q4_cols;
  q4_down_sel.columns = q4_rows;
  q4_down_sel.prompt = q4_down_prompt;
  q4_down_sel.staged = q4_down_staged;
  q4_down_sel.output = q4_down_output;
  TimingCtx q4_down_tile8 = q4_down_sel;
  q4_down_tile8.tile = 8;
  const float q4_down_sel_ms =
      time_launch(kWarmups, kSamples, timing_body, &q4_down_sel, &timing_error);
  const float q4_down_tile8_ms =
      time_launch(kWarmups, kSamples, timing_body, &q4_down_tile8,
                  &timing_error);
  const bool q8_faster =
      timing_error == cudaSuccess && q8_64_prod < q8_64_ref &&
      q8_256_prod < q8_256_ref;
  const bool q4_not_slower =
      timing_error == cudaSuccess &&
      (q4_sel_ms + q4_down_sel_ms) <= (q4_tile8_ms + q4_down_tile8_ms);

  cudaDeviceProp prop{};
  int device = 0;
  int driver = 0;
  int runtime = 0;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  cudaDriverGetVersion(&driver);
  cudaRuntimeGetVersion(&runtime);
  std::time_t now = std::time(nullptr);
  char utc[32]{};
  std::tm tm{};
  gmtime_r(&now, &tm);
  std::strftime(utc, sizeof(utc), "%Y-%m-%dT%H:%M:%SZ", &tm);

  const bool passed =
      q8_exact && q4_identical && q6_identical && envelope_ok && staging_ok &&
      invalid && q8_grid && q8_ref_grid && q4_grid && q6_grid && occupancy_ok &&
      q8_faster && q4_not_slower && q8_quality_grid && skinny_launch_ok;
  std::printf(
      "QW38_PROMPT_MMQ_RESULT={\"schema_version\":1,\"task\":\"OPT-009\","
      "\"status\":\"%s\",\"device\":\"%s\",\"compute_capability\":\"%d.%d\","
      "\"driver\":\"%d.%d\",\"runtime\":\"%d.%d\",\"toolkit\":\"CUDA 13.0.2\","
      "\"pinned_image\":\"qw38-cuda:13.0.2\",\"measurement_utc\":\"%s\","
      "\"threads\":256,\"output_rows_per_block\":8,"
      "\"semantic\":{\"q8_production_reference_exact\":%s,"
      "\"q4k_tiles_byte_identical\":%s,\"q6k_tiles_byte_identical\":%s,"
      "\"q4k_q6k_ds4_association\":%s,\"q8_staging_exact\":%s,"
      "\"invalid_input_rejected\":%s,\"q8_grid_reuses_weight_tiles\":%s,"
      "\"q8_reference_grid_is_row_wise\":%s,"
      "\"q4k_q6k_grid_matches_selected_tile\":%s,\"occupancy_admitted\":%s,"
      "\"q8_64_faster_than_reference\":%s,\"q8_256_faster_than_reference\":%s,"
      "\"q4k_4096_selected_not_slower_than_tile8\":%s},"
      "\"selected_tiles\":{\"q8_0\":{\"1\":%u,\"2\":%u,\"4\":%u,\"8\":%u,"
      "\"16\":%u,\"32\":%u,\"64\":%u,\"256\":%u,\"4096\":%u},"
      "\"q4_k\":{\"1\":%u,\"2\":%u,\"4\":%u,\"8\":%u,\"16\":%u,\"32\":%u,"
      "\"64\":%u,\"256\":%u,\"4096\":%u},"
      "\"q6_k\":{\"1\":%u,\"2\":%u,\"4\":%u,\"8\":%u,\"16\":%u,\"32\":%u,"
      "\"64\":%u,\"256\":%u,\"4096\":%u}},"
      "\"launches\":["
      "{\"kind\":\"q8_0\",\"prompt_rows\":64,\"kernel_nodes\":%d,"
      "\"grid\":[%u,%u,1],\"block\":[256,1,1],\"selected_tile\":%u},"
      "{\"kind\":\"q8_0\",\"prompt_rows\":4096,\"kernel_nodes\":%d,"
      "\"grid\":[%u,%u,1],\"block\":[256,1,1],\"selected_tile\":%u},"
      "{\"kind\":\"q8_0_reference\",\"prompt_rows\":64,\"kernel_nodes\":%d,"
      "\"grid\":[%u,%u,1],\"block\":[256,1,1],\"selected_tile\":1},"
      "{\"kind\":\"q8_0_reference\",\"prompt_rows\":4096,\"kernel_nodes\":%d,"
      "\"grid\":[%u,%u,1],\"block\":[256,1,1],\"selected_tile\":1},"
      "{\"kind\":\"q4_k\",\"prompt_rows\":64,\"kernel_nodes\":%d,"
      "\"grid\":[%u,%u,1],\"block\":[%u,%u,1],\"selected_tile\":%u},"
      "{\"kind\":\"q4_k\",\"prompt_rows\":4096,\"kernel_nodes\":%d,"
      "\"grid\":[%u,%u,1],\"block\":[%u,%u,1],\"selected_tile\":%u},"
      "{\"kind\":\"q6_k\",\"prompt_rows\":64,\"kernel_nodes\":%d,"
      "\"grid\":[%u,%u,1],\"block\":[%u,%u,1],\"selected_tile\":%u},"
      "{\"kind\":\"q6_k\",\"prompt_rows\":4096,\"kernel_nodes\":%d,"
      "\"grid\":[%u,%u,1],\"block\":[%u,%u,1],\"selected_tile\":%u}],"
      "\"kernel_attributes\":{\"q8_0\":{\"registers\":%d,"
      "\"local_bytes_per_thread\":%d,\"active_blocks_per_sm\":%d},"
      "\"q4_k\":{\"registers\":%d,\"local_bytes_per_thread\":%d,"
      "\"active_blocks_per_sm\":%d},"
      "\"q6_k\":{\"registers\":%d,\"local_bytes_per_thread\":%d,"
      "\"active_blocks_per_sm\":%d}},"
      "\"timing\":{\"q8_0_12288x5120_64\":{\"production_ms\":%.9g,"
      "\"reference_ms\":%.9g},\"q8_0_12288x5120_256\":{\"production_ms\":%.9g,"
      "\"reference_ms\":%.9g},\"q4_k_17408x5120_4096\":{\"selected_ms\":%.9g,"
      "\"tile8_ms\":%.9g},\"q4_k_5120x17408_4096\":{\"selected_ms\":%.9g,"
      "\"tile8_ms\":%.9g},\"q4_k_joint_4096\":{\"selected_ms\":%.9g,"
      "\"tile8_ms\":%.9g}},"
      "\"proof_limit\":\"component-only exact semantic, launch, occupancy, and "
      "tile-timing evidence; no end-to-end performance or speedup claim\"}\n",
      passed ? "measured" : "failed", prop.name, prop.major, prop.minor,
      driver / 1000, (driver % 1000) / 10, runtime / 1000,
      (runtime % 1000) / 10, utc, q8_exact ? "true" : "false",
      q4_identical ? "true" : "false", q6_identical ? "true" : "false",
      envelope_ok ? "true" : "false", staging_ok ? "true" : "false",
      invalid ? "true" : "false", q8_grid ? "true" : "false",
      q8_ref_grid ? "true" : "false",
      (q4_grid && q6_grid) ? "true" : "false",
      occupancy_ok ? "true" : "false", q8_faster ? "true" : "false",
      q8_faster ? "true" : "false", q4_not_slower ? "true" : "false",
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ8_0, 1),
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ8_0, 2),
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ8_0, 4),
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ8_0, 8),
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ8_0, 16),
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ8_0, 32),
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ8_0, 64),
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ8_0, 256),
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ8_0, 4096),
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ4K, 1),
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ4K, 2),
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ4K, 4),
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ4K, 8),
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ4K, 16),
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ4K, 32),
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ4K, 64),
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ4K, 256),
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ4K, 4096),
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ6K, 1),
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ6K, 2),
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ6K, 4),
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ6K, 8),
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ6K, 16),
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ6K, 32),
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ6K, 64),
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ6K, 256),
      qw38::cuda::selected_mmq_prompt_tile(qw38::cuda::QuantKind::kQ6K, 4096),
      q8_prod_64.nodes, q8_prod_64.grid_x, q8_prod_64.grid_y,
      q8_tile_64, q8_prod_4096.nodes, q8_prod_4096.grid_x, q8_prod_4096.grid_y,
      q8_tile_4096, q8_ref_64.nodes, q8_ref_64.grid_x, q8_ref_64.grid_y,
      q8_ref_4096.nodes, q8_ref_4096.grid_x, q8_ref_4096.grid_y, q4_64.nodes,
      q4_64.grid_x, q4_64.grid_y, q4_64.block.x, q4_64.block.y, q4_tile_64,
      q4_4096.nodes, q4_4096.grid_x, q4_4096.grid_y, q4_4096.block.x,
      q4_4096.block.y, q4_tile_4096, q6_64.nodes, q6_64.grid_x, q6_64.grid_y,
      q6_64.block.x, q6_64.block.y, q6_tile_64, q6_4096.nodes, q6_4096.grid_x,
      q6_4096.grid_y, q6_4096.block.x, q6_4096.block.y, q6_tile_4096,
      q8_prod_64.registers, q8_prod_64.local_bytes, q8_prod_64.active_blocks,
      q4_64.registers, q4_64.local_bytes, q4_64.active_blocks,
      q6_64.registers, q6_64.local_bytes, q6_64.active_blocks, q8_64_prod,
      q8_64_ref, q8_256_prod, q8_256_ref, q4_sel_ms, q4_tile8_ms,
      q4_down_sel_ms, q4_down_tile8_ms, q4_sel_ms + q4_down_sel_ms,
      q4_tile8_ms + q4_down_tile8_ms);

  cudaFree(q6_output);
  cudaFree(q6_staged);
  cudaFree(q6_prompt);
  cudaFree(q6_weights);
  cudaFree(q4_down_output);
  cudaFree(q4_down_staged);
  cudaFree(q4_down_prompt);
  cudaFree(q4_down_weights);
  cudaFree(q4_output);
  cudaFree(q4_staged);
  cudaFree(q4_prompt);
  cudaFree(q4_weights);
  cudaFree(q8_output);
  cudaFree(q8_staged);
  cudaFree(q8_prompt);
  cudaFree(q8_weights);
  return passed ? 0 : 1;
}
