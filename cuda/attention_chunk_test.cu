#include "attention_decode.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int fail_cuda(const char* operation, cudaError_t error) {
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
  return 1;
}

bool bf16_equal(const std::vector<__nv_bfloat16>& left,
                const std::vector<__nv_bfloat16>& right) {
  return left.size() == right.size() &&
         std::memcmp(left.data(), right.data(),
                     left.size() * sizeof(left[0])) == 0;
}

struct DeviceBuffers {
  float* query = nullptr;
  float* key = nullptr;
  float* value = nullptr;
  float* gate = nullptr;
  float* query_scale = nullptr;
  float* key_scale = nullptr;
  float* normalized_query = nullptr;
  float* normalized_key = nullptr;
  float* scores = nullptr;
  float* output = nullptr;
  __nv_bfloat16* committed_key = nullptr;
  __nv_bfloat16* committed_value = nullptr;
  __nv_bfloat16* candidate_key = nullptr;
  __nv_bfloat16* candidate_value = nullptr;
  std::uint64_t* frontier = nullptr;
};

void release(DeviceBuffers* buffers) {
  cudaFree(buffers->frontier);
  cudaFree(buffers->candidate_value);
  cudaFree(buffers->candidate_key);
  cudaFree(buffers->committed_value);
  cudaFree(buffers->committed_key);
  cudaFree(buffers->output);
  cudaFree(buffers->scores);
  cudaFree(buffers->normalized_key);
  cudaFree(buffers->normalized_query);
  cudaFree(buffers->key_scale);
  cudaFree(buffers->query_scale);
  cudaFree(buffers->gate);
  cudaFree(buffers->value);
  cudaFree(buffers->key);
  cudaFree(buffers->query);
}

cudaError_t allocate(const qw38::cuda::AttentionConfig& config,
                     std::size_t start_position, std::size_t token_count,
                     std::size_t candidate_tokens, DeviceBuffers* buffers) {
  const std::size_t query_values =
      qw38::cuda::attention_query_values(config);
  const std::size_t row_values =
      qw38::cuda::attention_kv_row_values(config);
  const std::size_t cache_values =
      qw38::cuda::attention_cache_values(config);
  const std::size_t score_values = qw38::cuda::attention_chunk_score_values(
      config, start_position, token_count);
  cudaError_t error = cudaMalloc(&buffers->query,
                                 token_count * query_values * sizeof(float));
#define QW38_ALLOC(pointer, count)                                            \
  if (error == cudaSuccess)                                                   \
  error = cudaMalloc(&(pointer), (count) * sizeof(*(pointer)))
  QW38_ALLOC(buffers->key, token_count * row_values);
  QW38_ALLOC(buffers->value, token_count * row_values);
  QW38_ALLOC(buffers->gate, token_count * query_values);
  QW38_ALLOC(buffers->query_scale, config.head_width);
  QW38_ALLOC(buffers->key_scale, config.head_width);
  QW38_ALLOC(buffers->normalized_query, query_values);
  QW38_ALLOC(buffers->normalized_key, row_values);
  QW38_ALLOC(buffers->scores, score_values);
  QW38_ALLOC(buffers->output, token_count * query_values);
  QW38_ALLOC(buffers->committed_key, cache_values);
  QW38_ALLOC(buffers->committed_value, cache_values);
  QW38_ALLOC(buffers->candidate_key, candidate_tokens * row_values);
  QW38_ALLOC(buffers->candidate_value, candidate_tokens * row_values);
  QW38_ALLOC(buffers->frontier, 1);
#undef QW38_ALLOC
  return error;
}

int run_chunk_case(const char* name,
                   const qw38::cuda::AttentionConfig& config,
                   std::size_t start_position, std::size_t token_count) {
  const std::size_t query_values =
      qw38::cuda::attention_query_values(config);
  const std::size_t row_values =
      qw38::cuda::attention_kv_row_values(config);
  const std::size_t cache_values =
      qw38::cuda::attention_cache_values(config);
  const std::size_t score_values = qw38::cuda::attention_chunk_score_values(
      config, start_position, token_count);
  std::vector<float> query(token_count * query_values);
  std::vector<float> key(token_count * row_values);
  std::vector<float> value(token_count * row_values);
  std::vector<float> gate(token_count * query_values);
  std::vector<float> query_scale(config.head_width);
  std::vector<float> key_scale(config.head_width);
  for (std::size_t index = 0; index < query.size(); ++index) {
    query[index] = std::sin(static_cast<float>(index) * 0.019F) * 0.75F;
    gate[index] =
        static_cast<float>(static_cast<int>(index % 19) - 9) * 0.046875F;
  }
  for (std::size_t index = 0; index < key.size(); ++index) {
    key[index] = std::cos(static_cast<float>(index) * 0.027F) * 0.625F;
    value[index] =
        static_cast<float>(static_cast<int>(index % 31) - 15) * 0.03125F;
  }
  for (std::size_t lane = 0; lane < config.head_width; ++lane) {
    query_scale[lane] = 0.875F + static_cast<float>(lane % 9) * 0.03125F;
    key_scale[lane] = 0.9375F + static_cast<float>(lane % 7) * 0.015625F;
  }
  std::vector<__nv_bfloat16> initial_key(cache_values);
  std::vector<__nv_bfloat16> initial_value(cache_values);
  for (std::size_t index = 0; index < cache_values; ++index) {
    initial_key[index] = __float2bfloat16_rn(
        static_cast<float>(static_cast<int>(index % 37) - 18) * 0.015625F);
    initial_value[index] = __float2bfloat16_rn(
        static_cast<float>(static_cast<int>(index % 41) - 20) * 0.015625F);
  }

  DeviceBuffers chunk;
  DeviceBuffers tokenwise;
  cudaError_t error =
      allocate(config, start_position, token_count, token_count, &chunk);
  if (error == cudaSuccess) {
    error = allocate(config, start_position, token_count, 1, &tokenwise);
  }
  if (error != cudaSuccess) return fail_cuda("chunk cudaMalloc", error);
#define QW38_COPY(pointer, source)                                            \
  if (error == cudaSuccess)                                                   \
  error = cudaMemcpy((pointer), (source).data(),                              \
                     (source).size() * sizeof((source)[0]), cudaMemcpyHostToDevice)
  QW38_COPY(chunk.query, query);
  QW38_COPY(chunk.key, key);
  QW38_COPY(chunk.value, value);
  QW38_COPY(chunk.gate, gate);
  QW38_COPY(chunk.query_scale, query_scale);
  QW38_COPY(chunk.key_scale, key_scale);
  QW38_COPY(chunk.committed_key, initial_key);
  QW38_COPY(chunk.committed_value, initial_value);
  QW38_COPY(tokenwise.query, query);
  QW38_COPY(tokenwise.key, key);
  QW38_COPY(tokenwise.value, value);
  QW38_COPY(tokenwise.gate, gate);
  QW38_COPY(tokenwise.query_scale, query_scale);
  QW38_COPY(tokenwise.key_scale, key_scale);
  QW38_COPY(tokenwise.committed_key, initial_key);
  QW38_COPY(tokenwise.committed_value, initial_value);
#undef QW38_COPY
  const std::uint64_t initial_frontier = start_position;
  if (error == cudaSuccess) {
    error = cudaMemcpy(chunk.frontier, &initial_frontier,
                       sizeof(initial_frontier), cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(tokenwise.frontier, &initial_frontier,
                       sizeof(initial_frontier), cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) return fail_cuda("chunk cudaMemcpy H2D", error);
  const qw38::cuda::AttentionCache chunk_committed{
      chunk.committed_key, chunk.committed_value};
  const qw38::cuda::AttentionCache chunk_candidate{chunk.candidate_key,
                                                    chunk.candidate_value};
  for (int warmup = 0; warmup < 3 && error == cudaSuccess; ++warmup) {
    error = qw38::cuda::launch_attention_prepare_chunk(
        config, start_position, token_count, chunk.query, chunk.key, chunk.value,
        chunk.query_scale, chunk.key_scale, chunk.gate, chunk_committed,
        chunk_candidate, chunk.normalized_query, chunk.normalized_key,
        chunk.scores, chunk.output, nullptr);
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (error == cudaSuccess) error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error == cudaSuccess) error = cudaEventRecord(start);
  for (int sample = 0; sample < 30 && error == cudaSuccess; ++sample) {
    error = qw38::cuda::launch_attention_prepare_chunk(
        config, start_position, token_count, chunk.query, chunk.key, chunk.value,
        chunk.query_scale, chunk.key_scale, chunk.gate, chunk_committed,
        chunk_candidate, chunk.normalized_query, chunk.normalized_key,
        chunk.scores, chunk.output, nullptr);
  }
  if (error == cudaSuccess) error = cudaEventRecord(stop);
  if (error == cudaSuccess) error = cudaEventSynchronize(stop);
  if (error != cudaSuccess) return fail_cuda("chunk prepare", error);
  float milliseconds = 0.0F;
  error = cudaEventElapsedTime(&milliseconds, start, stop);
  if (error != cudaSuccess) return fail_cuda("chunk timing", error);

  std::vector<__nv_bfloat16> actual_committed_key(cache_values);
  std::vector<__nv_bfloat16> actual_committed_value(cache_values);
#define QW38_READ(destination, pointer)                                       \
  if (error == cudaSuccess)                                                   \
  error = cudaMemcpy((destination).data(), (pointer),                         \
                     (destination).size() * sizeof((destination)[0]),         \
                     cudaMemcpyDeviceToHost)
  QW38_READ(actual_committed_key, chunk.committed_key);
  QW38_READ(actual_committed_value, chunk.committed_value);
  std::uint64_t actual_frontier = 0;
  if (error == cudaSuccess) {
    error = cudaMemcpy(&actual_frontier, chunk.frontier,
                       sizeof(actual_frontier), cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) return fail_cuda("chunk atomic read", error);
  const bool prepare_atomic = bf16_equal(actual_committed_key, initial_key) &&
                              bf16_equal(actual_committed_value, initial_value) &&
                              actual_frontier == initial_frontier;

  const qw38::cuda::AttentionCache token_committed{
      tokenwise.committed_key, tokenwise.committed_value};
  const qw38::cuda::AttentionCache token_candidate{tokenwise.candidate_key,
                                                    tokenwise.candidate_value};
  for (std::size_t token = 0; token < token_count && error == cudaSuccess;
       ++token) {
    error = qw38::cuda::launch_attention_prepare(
        config, start_position + token,
        tokenwise.query + token * query_values,
        tokenwise.key + token * row_values,
        tokenwise.value + token * row_values, tokenwise.query_scale,
        tokenwise.key_scale, tokenwise.gate + token * query_values,
        token_committed, token_candidate, tokenwise.normalized_query,
        tokenwise.normalized_key, tokenwise.scores,
        tokenwise.output + token * query_values, nullptr);
    if (error == cudaSuccess) {
      error = qw38::cuda::launch_attention_commit(
          config, start_position + token, token_candidate, token_committed,
          start_position + token + 1, tokenwise.frontier, nullptr);
    }
  }
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return fail_cuda("token-wise attention", error);
  std::vector<float> chunk_output(token_count * query_values);
  std::vector<float> token_output(token_count * query_values);
  std::vector<__nv_bfloat16> chunk_candidate_key(token_count * row_values);
  std::vector<__nv_bfloat16> chunk_candidate_value(token_count * row_values);
  QW38_READ(chunk_output, chunk.output);
  QW38_READ(token_output, tokenwise.output);
  QW38_READ(chunk_candidate_key, chunk.candidate_key);
  QW38_READ(chunk_candidate_value, chunk.candidate_value);
  QW38_READ(actual_committed_key, tokenwise.committed_key);
  QW38_READ(actual_committed_value, tokenwise.committed_value);
  if (error != cudaSuccess) return fail_cuda("token-wise read", error);
  bool tokenwise_equal = chunk_output == token_output;
  for (std::size_t token = 0; token < token_count; ++token) {
    const std::size_t cache_base = (start_position + token) * row_values;
    const std::size_t candidate_base = token * row_values;
    tokenwise_equal =
        tokenwise_equal &&
        std::memcmp(chunk_candidate_key.data() + candidate_base,
                    actual_committed_key.data() + cache_base,
                    row_values * sizeof(chunk_candidate_key[0])) == 0 &&
        std::memcmp(chunk_candidate_value.data() + candidate_base,
                    actual_committed_value.data() + cache_base,
                    row_values * sizeof(chunk_candidate_value[0])) == 0;
  }
  error = qw38::cuda::launch_attention_commit_chunk(
      config, start_position, token_count, chunk_candidate, chunk_committed,
      start_position + token_count, chunk.frontier, nullptr);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return fail_cuda("chunk commit", error);
  std::vector<__nv_bfloat16> chunk_final_key(cache_values);
  std::vector<__nv_bfloat16> chunk_final_value(cache_values);
  QW38_READ(chunk_final_key, chunk.committed_key);
  QW38_READ(chunk_final_value, chunk.committed_value);
#undef QW38_READ
  if (error == cudaSuccess) {
    error = cudaMemcpy(&actual_frontier, chunk.frontier,
                       sizeof(actual_frontier), cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) return fail_cuda("chunk final read", error);
  const bool commit_exact =
      bf16_equal(chunk_final_key, actual_committed_key) &&
      bf16_equal(chunk_final_value, actual_committed_value) &&
      actual_frontier == start_position + token_count;
  std::size_t nonfinite = 0;
  for (float item : chunk_output) {
    if (!std::isfinite(item)) ++nonfinite;
  }
  std::printf(
      "attention_chunk=%s start=%zu tokens=%zu score_values=%zu "
      "quadratic_values=%zu nonfinite=%zu prepare_atomic=%s "
      "tokenwise_equal=%s commit_exact=%s mean_ms=%.9g\n",
      name, start_position, token_count, score_values,
      token_count * score_values, nonfinite,
      prepare_atomic ? "true" : "false",
      tokenwise_equal ? "true" : "false", commit_exact ? "true" : "false",
      milliseconds / 30.0F);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  release(&tokenwise);
  release(&chunk);
  return nonfinite == 0 && prepare_atomic && tokenwise_equal && commit_exact
             ? 0
             : 1;
}

int run_capacity_case() {
  const qw38::cuda::AttentionConfig config{24, 4, 256, 64, 131072};
  const std::size_t position = 131071;
  const std::size_t query_values =
      qw38::cuda::attention_query_values(config);
  const std::size_t row_values =
      qw38::cuda::attention_kv_row_values(config);
  const std::size_t cache_values =
      qw38::cuda::attention_cache_values(config);
  const std::size_t score_values =
      qw38::cuda::attention_score_values(config, position);
  DeviceBuffers buffers;
  cudaError_t error = allocate(config, position, 1, 1, &buffers);
  if (error != cudaSuccess) return fail_cuda("capacity cudaMalloc", error);
  error = cudaMemset(buffers.committed_key, 0,
                     cache_values * sizeof(__nv_bfloat16));
  if (error == cudaSuccess) {
    error = cudaMemset(buffers.committed_value, 0,
                       cache_values * sizeof(__nv_bfloat16));
  }
  std::vector<float> query(query_values);
  std::vector<float> key(row_values);
  std::vector<float> value(row_values);
  std::vector<float> gate(query_values);
  std::vector<float> scale(config.head_width, 1.0F);
  for (std::size_t index = 0; index < query_values; ++index) {
    query[index] = std::sin(static_cast<float>(index) * 0.011F) * 0.5F;
    gate[index] = 0.0F;
  }
  for (std::size_t index = 0; index < row_values; ++index) {
    key[index] = std::cos(static_cast<float>(index) * 0.013F) * 0.5F;
    value[index] =
        static_cast<float>(static_cast<int>(index % 17) - 8) * 0.03125F;
  }
#define QW38_COPY(pointer, source)                                            \
  if (error == cudaSuccess)                                                   \
  error = cudaMemcpy((pointer), (source).data(),                              \
                     (source).size() * sizeof((source)[0]), cudaMemcpyHostToDevice)
  QW38_COPY(buffers.query, query);
  QW38_COPY(buffers.key, key);
  QW38_COPY(buffers.value, value);
  QW38_COPY(buffers.gate, gate);
  QW38_COPY(buffers.query_scale, scale);
  QW38_COPY(buffers.key_scale, scale);
#undef QW38_COPY
  const std::uint64_t initial_frontier = position;
  if (error == cudaSuccess) {
    error = cudaMemcpy(buffers.frontier, &initial_frontier,
                       sizeof(initial_frontier), cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) return fail_cuda("capacity initialization", error);
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  error = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (error != cudaSuccess) return fail_cuda("capacity mem info", error);
  const qw38::cuda::AttentionCache committed{buffers.committed_key,
                                              buffers.committed_value};
  const qw38::cuda::AttentionCache candidate{buffers.candidate_key,
                                              buffers.candidate_value};
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error == cudaSuccess) error = cudaEventRecord(start);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_attention_prepare(
        config, position, buffers.query, buffers.key, buffers.value,
        buffers.query_scale, buffers.key_scale, buffers.gate, committed,
        candidate, buffers.normalized_query, buffers.normalized_key,
        buffers.scores, buffers.output, nullptr);
  }
  if (error == cudaSuccess) error = cudaEventRecord(stop);
  if (error == cudaSuccess) error = cudaEventSynchronize(stop);
  if (error != cudaSuccess) return fail_cuda("capacity prepare", error);
  float milliseconds = 0.0F;
  error = cudaEventElapsedTime(&milliseconds, start, stop);
  if (error != cudaSuccess) return fail_cuda("capacity timing", error);
  std::vector<float> output(query_values);
  error = cudaMemcpy(output.data(), buffers.output,
                     output.size() * sizeof(output[0]), cudaMemcpyDeviceToHost);
  std::uint64_t actual_frontier = 0;
  if (error == cudaSuccess) {
    error = cudaMemcpy(&actual_frontier, buffers.frontier,
                       sizeof(actual_frontier), cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) return fail_cuda("capacity read", error);
  std::size_t nonfinite = 0;
  for (float item : output) {
    if (!std::isfinite(item)) ++nonfinite;
  }
  const bool prepare_atomic = actual_frontier == initial_frontier;
  error = qw38::cuda::launch_attention_commit(
      config, position, candidate, committed, config.capacity,
      buffers.frontier, nullptr);
  if (error == cudaSuccess) error = cudaDeviceSynchronize();
  if (error == cudaSuccess) {
    error = cudaMemcpy(&actual_frontier, buffers.frontier,
                       sizeof(actual_frontier), cudaMemcpyDeviceToHost);
  }
  if (error != cudaSuccess) return fail_cuda("capacity commit", error);
  const bool commit_exact = actual_frontier == config.capacity;
  const bool overflow_rejected =
      qw38::cuda::attention_chunk_score_values(config, config.capacity, 1) == 0 &&
      qw38::cuda::attention_chunk_score_values(config, position, 2) == 0 &&
      qw38::cuda::launch_attention_prepare_chunk(
          config, position, 2, buffers.query, buffers.key, buffers.value,
          buffers.query_scale, buffers.key_scale, buffers.gate, committed,
          candidate, buffers.normalized_query, buffers.normalized_key,
          buffers.scores, buffers.output, nullptr) == cudaErrorInvalidValue;
  const std::size_t cache_bytes =
      2 * cache_values * sizeof(__nv_bfloat16);
  const std::size_t score_bytes = score_values * sizeof(float);
  std::printf(
      "attention_capacity=production capacity=%u position=%zu "
      "cache_bytes=%zu score_bytes=%zu free_after_alloc=%zu total_bytes=%zu "
      "nonfinite=%zu prepare_atomic=%s commit_exact=%s overflow_rejected=%s "
      "elapsed_ms=%.9g\n",
      config.capacity, position, cache_bytes, score_bytes, free_bytes,
      total_bytes, nonfinite, prepare_atomic ? "true" : "false",
      commit_exact ? "true" : "false",
      overflow_rejected ? "true" : "false", milliseconds);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  release(&buffers);
  return nonfinite == 0 && prepare_atomic && commit_exact && overflow_rejected
             ? 0
             : 1;
}

}  // namespace

int main() {
  const qw38::cuda::AttentionConfig small{6, 2, 8, 4, 16};
  const qw38::cuda::AttentionConfig production{24, 4, 256, 64, 16};
  if (qw38::cuda::attention_chunk_score_values(small, 2, 3) != 30 ||
      qw38::cuda::attention_chunk_score_values(small, 2, 0) != 0 ||
      run_chunk_case("small_3", small, 2, 3) != 0 ||
      run_chunk_case("small_9", small, 2, 9) != 0 ||
      run_chunk_case("production_9", production, 2, 9) != 0 ||
      run_capacity_case() != 0) {
    return 1;
  }
  std::printf("status=passed\n");
  return 0;
}
