#include "attention_decode.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

namespace {
using qw38::cuda::AttentionCache;
using qw38::cuda::AttentionConfig;
using qw38::cuda::AttentionKvTileSpan;

struct Buffers {
  float *q{}, *k{}, *v{}, *gate{}, *qs{}, *ks{}, *nq{}, *nk{}, *score{},
      *out{};
  __nv_bfloat16 *ck{}, *cv{}, *tk{}, *tv{};
};

void release(Buffers& b) {
  cudaFree(b.tv);
  cudaFree(b.tk);
  cudaFree(b.cv);
  cudaFree(b.ck);
  cudaFree(b.out);
  cudaFree(b.score);
  cudaFree(b.nk);
  cudaFree(b.nq);
  cudaFree(b.ks);
  cudaFree(b.qs);
  cudaFree(b.gate);
  cudaFree(b.v);
  cudaFree(b.k);
  cudaFree(b.q);
}

bool allocate(Buffers& b, const AttentionConfig& c, std::size_t rows,
              std::size_t start) {
  const std::size_t q = qw38::cuda::attention_query_values(c);
  const std::size_t r = qw38::cuda::attention_kv_row_values(c);
  const std::size_t cache = qw38::cuda::attention_cache_values(c);
  const std::size_t scores =
      qw38::cuda::attention_chunk_score_values(c, start, rows);
#define ALLOCATE(field, count)                                            \
  if (cudaMalloc(reinterpret_cast<void**>(&b.field),                      \
                 (count) * sizeof(*b.field)) != cudaSuccess)              \
    return false
  ALLOCATE(q, rows * q);
  ALLOCATE(k, rows * r);
  ALLOCATE(v, rows * r);
  ALLOCATE(gate, rows * q);
  ALLOCATE(qs, c.head_width);
  ALLOCATE(ks, c.head_width);
  ALLOCATE(nq, q);
  ALLOCATE(nk, r);
  ALLOCATE(score, scores);
  ALLOCATE(out, rows * q);
  ALLOCATE(ck, cache);
  ALLOCATE(cv, cache);
  ALLOCATE(tk, rows * r);
  ALLOCATE(tv, rows * r);
#undef ALLOCATE
  return true;
}

void seed_logical_pattern(std::vector<__nv_bfloat16>* logical,
                          std::size_t values) {
  logical->resize(values);
  for (std::size_t i = 0; i < values; ++i) {
    (*logical)[i] = __float2bfloat16_rn(
        static_cast<float>(static_cast<int>(i % 31) - 15) * .015625F);
  }
}

void seed(Buffers& b, const AttentionConfig& c, std::size_t rows,
          std::size_t start) {
  const std::size_t q = qw38::cuda::attention_query_values(c);
  const std::size_t r = qw38::cuda::attention_kv_row_values(c);
  const std::size_t cache = qw38::cuda::attention_cache_values(c);
  std::vector<float> hq(rows * q), hk(rows * r), hv(rows * r), hg(rows * q),
      scale(c.head_width, 1.0F);
  for (std::size_t i = 0; i < hq.size(); ++i) {
    hq[i] = sinf(static_cast<float>(i) * .001F);
    hg[i] = cosf(static_cast<float>(i) * .002F);
  }
  for (std::size_t i = 0; i < hk.size(); ++i) {
    hk[i] = cosf(static_cast<float>(i) * .003F);
    hv[i] = sinf(static_cast<float>(i) * .004F);
  }
  const std::size_t prefix_tokens = start == 0 ? 0 : start;
  std::vector<__nv_bfloat16> logical;
  seed_logical_pattern(&logical, prefix_tokens * r);
  std::vector<__nv_bfloat16> physical(cache);
  std::memset(physical.data(), 0, physical.size() * sizeof(physical[0]));
  if (prefix_tokens != 0) {
    qw38::cuda::attention_kv_scatter_logical_rows(
        logical.data(), physical.data(), 0, prefix_tokens, c.kv_heads,
        c.capacity, c.head_width);
  }
  cudaMemcpy(b.q, hq.data(), hq.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b.k, hk.data(), hk.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b.v, hv.data(), hv.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b.gate, hg.data(), hg.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b.qs, scale.data(), scale.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(b.ks, scale.data(), scale.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(b.ck, physical.data(), physical.size() * sizeof(physical[0]),
             cudaMemcpyHostToDevice);
  cudaMemcpy(b.cv, physical.data(), physical.size() * sizeof(physical[0]),
             cudaMemcpyHostToDevice);
  cudaMemset(b.score, 0xA5,
             qw38::cuda::attention_chunk_score_values(c, start, rows) *
                 sizeof(float));
}

bool bijective(std::uint32_t kv_heads, std::uint32_t capacity,
               std::uint32_t head_width) {
  const std::size_t values =
      static_cast<std::size_t>(kv_heads) * head_width * capacity;
  std::vector<unsigned char> seen(values, 0);
  for (std::size_t token = 0; token < capacity; ++token) {
    for (std::uint32_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
      for (std::uint32_t lane = 0; lane < head_width; ++lane) {
        const std::size_t index = qw38::cuda::attention_kv_physical_index(
            token, kv_head, lane, capacity, head_width);
        if (index >= values || seen[index] != 0) return false;
        seen[index] = 1;
      }
    }
  }
  for (unsigned char flag : seen) {
    if (flag == 0) return false;
  }
  return true;
}

bool inspect_launch(const AttentionConfig& c, Buffers& b, int* nodes,
                    dim3* staging_grid, dim3* staging_block,
                    dim3* attention_grid, dim3* attention_block,
                    unsigned* dynamic_shared) {
  cudaStream_t stream{};
  cudaGraph_t graph{};
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
    return false;
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  AttentionCache committed{b.ck, b.cv}, candidate{b.tk, b.tv};
  cudaError_t error = qw38::cuda::launch_attention_prepare_chunk(
      c, 0, 64, b.q, b.k, b.v, b.qs, b.ks, b.gate, committed, candidate, b.nq,
      b.nk, b.score, b.out, stream);
  if (error == cudaSuccess) error = cudaStreamEndCapture(stream, &graph);
  std::size_t count = 0;
  if (error == cudaSuccess) error = cudaGraphGetNodes(graph, nullptr, &count);
  std::vector<cudaGraphNode_t> graph_nodes(count);
  if (error == cudaSuccess)
    error = cudaGraphGetNodes(graph, graph_nodes.data(), &count);
  *nodes = static_cast<int>(count);
  for (cudaGraphNode_t node : graph_nodes) {
    cudaGraphNodeType type{};
    cudaKernelNodeParams params{};
    if (cudaGraphNodeGetType(node, &type) != cudaSuccess ||
        type != cudaGraphNodeTypeKernel ||
        cudaGraphKernelNodeGetParams(node, &params) != cudaSuccess) {
      error = cudaErrorInvalidValue;
      break;
    }
    if (params.sharedMemBytes == 0) {
      *staging_grid = params.gridDim;
      *staging_block = params.blockDim;
    } else {
      *attention_grid = params.gridDim;
      *attention_block = params.blockDim;
      *dynamic_shared = params.sharedMemBytes;
    }
  }
  if (graph != nullptr) cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  return error == cudaSuccess;
}

}  // namespace

int main() {
  const AttentionConfig production{24, 4, 256, 64, 131072};
  const AttentionConfig tiny{24, 4, 256, 64, 3};
  const std::size_t row_values = qw38::cuda::attention_kv_row_values(production);
  const std::size_t cache_values =
      qw38::cuda::attention_cache_values(production);
  const std::size_t sixteen_layer_bytes =
      16 * 2 * cache_values * sizeof(__nv_bfloat16);
  bool bijection_ok = bijective(production.kv_heads, production.capacity,
                                production.head_width) &&
                      bijective(tiny.kv_heads, tiny.capacity, tiny.head_width);

  const std::size_t round_prefixes[] = {1, 2, 31, 32, 33, 65};
  bool round_trip = true;
  {
    std::vector<__nv_bfloat16> logical;
    seed_logical_pattern(&logical, 65 * row_values);
    std::vector<__nv_bfloat16> physical(cache_values);
    std::memset(physical.data(), 0, physical.size() * sizeof(physical[0]));
    qw38::cuda::attention_kv_scatter_logical_rows(
        logical.data(), physical.data(), 0, 65, production.kv_heads,
        production.capacity, production.head_width);
    __nv_bfloat16 *device_physical = nullptr, *device_logical = nullptr;
    if (cudaMalloc(&device_physical, cache_values * sizeof(__nv_bfloat16)) !=
            cudaSuccess ||
        cudaMalloc(&device_logical, 65 * row_values * sizeof(__nv_bfloat16)) !=
            cudaSuccess)
      return 2;
    cudaMemcpy(device_physical, physical.data(),
               physical.size() * sizeof(physical[0]), cudaMemcpyHostToDevice);
    for (std::size_t prefix : round_prefixes) {
      if (qw38::cuda::launch_pack_committed_kv(
              production, 0, prefix, device_physical, device_logical,
              nullptr) != cudaSuccess ||
          cudaDeviceSynchronize() != cudaSuccess) {
        round_trip = false;
        break;
      }
      std::vector<__nv_bfloat16> packed(prefix * row_values);
      cudaMemcpy(packed.data(), device_logical,
                 packed.size() * sizeof(packed[0]), cudaMemcpyDeviceToHost);
      round_trip &= std::memcmp(packed.data(), logical.data(),
                                packed.size() * sizeof(packed[0])) == 0;
    }
    cudaFree(device_logical);
    cudaFree(device_physical);
  }

  const std::size_t span_prefixes[] = {0, 31, 32, 2048};
  bool tiles_coalesced = true;
  constexpr std::uint32_t kSpanCapacity = 65536;
  for (std::size_t start : span_prefixes) {
    Buffers buffers{};
    if (!allocate(buffers, production, 64, start)) return 2;
    seed(buffers, production, 64, start);
    AttentionKvTileSpan* device_spans = nullptr;
    std::uint32_t* device_count = nullptr;
    cudaMalloc(&device_spans, kSpanCapacity * sizeof(AttentionKvTileSpan));
    cudaMalloc(&device_count, sizeof(std::uint32_t));
    cudaMemset(device_count, 0, sizeof(std::uint32_t));
    AttentionCache committed{buffers.ck, buffers.cv};
    AttentionCache candidate{buffers.tk, buffers.tv};
    if (qw38::cuda::launch_attention_prepare_chunk_tile_span_instrumented(
            production, start, 64, buffers.q, buffers.k, buffers.v, buffers.qs,
            buffers.ks, buffers.gate, committed, candidate, buffers.nq,
            buffers.nk, buffers.score, buffers.out, device_spans, device_count,
            kSpanCapacity, nullptr) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess)
      return 3;
    std::uint32_t span_count = 0;
    cudaMemcpy(&span_count, device_count, sizeof(span_count),
               cudaMemcpyDeviceToHost);
    if (span_count == 0 || span_count > kSpanCapacity) {
      tiles_coalesced = false;
    } else {
      std::vector<AttentionKvTileSpan> spans(span_count);
      cudaMemcpy(spans.data(), device_spans,
                 span_count * sizeof(AttentionKvTileSpan),
                 cudaMemcpyDeviceToHost);
      bool saw_full = start >= qw38::cuda::kAttentionKvTileRows;
      bool found_full = false;
      for (const AttentionKvTileSpan& span : spans) {
        const bool full_committed =
            span.coalesced == 1 &&
            span.rows == qw38::cuda::kAttentionKvTileRows;
        if (full_committed) {
          found_full = true;
          const std::uint64_t expected =
              (static_cast<std::uint64_t>(span.rows) * production.head_width -
               1) *
              sizeof(__nv_bfloat16);
          tiles_coalesced &= span.last_address >= span.first_address &&
                             span.last_address - span.first_address == expected;
        }
      }
      if (saw_full) tiles_coalesced &= found_full;
    }
    cudaFree(device_count);
    cudaFree(device_spans);
    release(buffers);
  }

  const std::size_t row_cases[] = {1, 2, 3, 31, 32, 33, 64, 65};
  const std::size_t exact_prefixes[] = {0, 31, 32};
  const std::size_t qv = qw38::cuda::attention_query_values(production);
  bool exact = true, candidate_exact = true, finite = true, isolated = true,
       scratch = true;
  for (std::size_t start : exact_prefixes) {
    for (std::size_t rows : row_cases) {
      Buffers production_buffers{}, reference{};
      if (!allocate(production_buffers, production, rows, start) ||
          !allocate(reference, production, rows, start))
        return 2;
      seed(production_buffers, production, rows, start);
      seed(reference, production, rows, start);
      std::vector<__nv_bfloat16> cache_before(
          qw38::cuda::attention_cache_values(production) * 2);
      cudaMemcpy(cache_before.data(), production_buffers.ck,
                 cache_before.size() / 2 * sizeof(__nv_bfloat16),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(cache_before.data() + cache_before.size() / 2,
                 production_buffers.cv,
                 cache_before.size() / 2 * sizeof(__nv_bfloat16),
                 cudaMemcpyDeviceToHost);
      std::vector<unsigned char> score_before(
          qw38::cuda::attention_chunk_score_values(production, start, rows) *
          sizeof(float));
      cudaMemcpy(score_before.data(), production_buffers.score,
                 score_before.size(), cudaMemcpyDeviceToHost);
      std::uint64_t* dummy = nullptr;
      cudaMalloc(&dummy, sizeof(std::uint64_t));
      if (qw38::cuda::launch_attention_prepare_chunk(
              production, start, rows, production_buffers.q,
              production_buffers.k, production_buffers.v,
              production_buffers.qs, production_buffers.ks,
              production_buffers.gate,
              {production_buffers.ck, production_buffers.cv},
              {production_buffers.tk, production_buffers.tv},
              production_buffers.nq, production_buffers.nk,
              production_buffers.score, production_buffers.out,
              nullptr) != cudaSuccess ||
          qw38::cuda::launch_attention_prepare_chunk_grouped_instrumented(
              production, start, rows, reference.q, reference.k, reference.v,
              reference.qs, reference.ks, reference.gate,
              {reference.ck, reference.cv}, {reference.tk, reference.tv},
              reference.nq, reference.nk, reference.score, reference.out,
              dummy, nullptr) != cudaSuccess ||
          cudaDeviceSynchronize() != cudaSuccess)
        return 3;
      std::vector<float> po(rows * qv), ro(rows * qv);
      cudaMemcpy(po.data(), production_buffers.out, po.size() * sizeof(float),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(ro.data(), reference.out, ro.size() * sizeof(float),
                 cudaMemcpyDeviceToHost);
      exact &= std::memcmp(po.data(), ro.data(), po.size() * sizeof(float)) == 0;
      for (float value : po) finite &= std::isfinite(value);
      std::vector<__nv_bfloat16> pk(rows * row_values), rk(rows * row_values),
          pv(rows * row_values), rv(rows * row_values);
      cudaMemcpy(pk.data(), production_buffers.tk, pk.size() * sizeof(pk[0]),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(rk.data(), reference.tk, rk.size() * sizeof(rk[0]),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(pv.data(), production_buffers.tv, pv.size() * sizeof(pv[0]),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(rv.data(), reference.tv, rv.size() * sizeof(rv[0]),
                 cudaMemcpyDeviceToHost);
      candidate_exact &=
          std::memcmp(pk.data(), rk.data(), pk.size() * sizeof(pk[0])) == 0 &&
          std::memcmp(pv.data(), rv.data(), pv.size() * sizeof(pv[0])) == 0;
      std::vector<__nv_bfloat16> cache_after(cache_before.size());
      cudaMemcpy(cache_after.data(), production_buffers.ck,
                 cache_after.size() / 2 * sizeof(__nv_bfloat16),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(cache_after.data() + cache_after.size() / 2,
                 production_buffers.cv,
                 cache_after.size() / 2 * sizeof(__nv_bfloat16),
                 cudaMemcpyDeviceToHost);
      isolated &= cache_before == cache_after;
      std::vector<unsigned char> score_after(score_before.size());
      cudaMemcpy(score_after.data(), production_buffers.score,
                 score_after.size(), cudaMemcpyDeviceToHost);
      scratch &= score_before == score_after;
      cudaFree(dummy);
      release(production_buffers);
      release(reference);
    }
  }

  bool commit_pack = true, frontier_once = true;
  {
    const std::size_t start = 31;
    const std::size_t rows = 33;
    Buffers buffers{};
    if (!allocate(buffers, production, rows, start)) return 2;
    seed(buffers, production, rows, start);
    __nv_bfloat16* packed = nullptr;
    cudaMalloc(&packed, (start + rows) * row_values * sizeof(__nv_bfloat16));
    if (qw38::cuda::launch_pack_committed_kv(production, 0, start, buffers.ck,
                                             packed, nullptr) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess)
      return 3;
    std::vector<__nv_bfloat16> previous(start * row_values);
    cudaMemcpy(previous.data(), packed, previous.size() * sizeof(previous[0]),
               cudaMemcpyDeviceToHost);
    if (qw38::cuda::launch_attention_prepare_chunk(
            production, start, rows, buffers.q, buffers.k, buffers.v,
            buffers.qs, buffers.ks, buffers.gate, {buffers.ck, buffers.cv},
            {buffers.tk, buffers.tv}, buffers.nq, buffers.nk, buffers.score,
            buffers.out, nullptr) != cudaSuccess)
      return 3;
    std::uint64_t frontier_value = start, *frontier = nullptr;
    cudaMalloc(&frontier, sizeof(frontier_value));
    cudaMemcpy(frontier, &frontier_value, sizeof(frontier_value),
               cudaMemcpyHostToDevice);
    if (qw38::cuda::launch_attention_commit_chunk(
            production, start, rows, {buffers.tk, buffers.tv},
            {buffers.ck, buffers.cv}, start + rows, frontier,
            nullptr) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess)
      return 3;
    cudaMemcpy(&frontier_value, frontier, sizeof(frontier_value),
               cudaMemcpyDeviceToHost);
    frontier_once &= frontier_value == start + rows;
    if (qw38::cuda::launch_pack_committed_kv(production, 0, start + rows,
                                             buffers.ck, packed,
                                             nullptr) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess)
      return 3;
    std::vector<__nv_bfloat16> after((start + rows) * row_values);
    std::vector<__nv_bfloat16> candidate(rows * row_values);
    cudaMemcpy(after.data(), packed, after.size() * sizeof(after[0]),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(candidate.data(), buffers.tk,
               candidate.size() * sizeof(candidate[0]), cudaMemcpyDeviceToHost);
    commit_pack &=
        std::memcmp(after.data(), previous.data(),
                    previous.size() * sizeof(previous[0])) == 0 &&
        std::memcmp(after.data() + previous.size(), candidate.data(),
                    candidate.size() * sizeof(candidate[0])) == 0;
    cudaFree(frontier);
    cudaFree(packed);
    release(buffers);
  }

  Buffers graph{};
  if (!allocate(graph, production, 64, 0)) return 2;
  seed(graph, production, 64, 0);
  int nodes = 0;
  dim3 staging_grid{}, staging_block{}, attention_grid{}, attention_block{};
  unsigned dynamic_shared = 0;
  const bool launch_ok =
      inspect_launch(production, graph, &nodes, &staging_grid, &staging_block,
                     &attention_grid, &attention_block, &dynamic_shared) &&
      nodes == 2 && staging_grid.x == 4 && staging_grid.y == 64 &&
      staging_grid.z == 1 && attention_grid.x == 4 && attention_grid.y == 32 &&
      attention_grid.z == 1 && staging_block.x == 256 && staging_block.y == 1 &&
      staging_block.z == 1 && attention_block.x == 256 &&
      attention_block.y == 1 && attention_block.z == 1 &&
      dynamic_shared == 33792;
  release(graph);

  const bool invalid =
      qw38::cuda::launch_attention_prepare_chunk(
          production, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr,
          nullptr, {}, {}, nullptr, nullptr, nullptr, nullptr, nullptr) ==
      cudaErrorInvalidValue;

  if (!bijection_ok || !round_trip || !tiles_coalesced || !exact ||
      !candidate_exact || !finite || !isolated || !scratch || !commit_pack ||
      !frontier_once || !launch_ok || !invalid || cache_values != 134217728 ||
      sixteen_layer_bytes != 8589934592ULL)
    return 4;

  cudaDeviceProp prop{};
  int device = 0, driver = 0, runtime = 0;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  cudaDriverGetVersion(&driver);
  cudaRuntimeGetVersion(&runtime);
  std::time_t now = std::time(nullptr);
  char utc[32]{};
  std::tm tm{};
  gmtime_r(&now, &tm);
  std::strftime(utc, sizeof(utc), "%Y-%m-%dT%H:%M:%SZ", &tm);
  std::printf(
      "QW38_KV_TILE_LAYOUT_RESULT={\"schema_version\":1,\"task\":\"OPT-010\","
      "\"status\":\"measured\",\"device\":\"%s\",\"compute_capability\":\"%d.%d\","
      "\"driver\":\"%d.%d\",\"runtime\":\"%d.%d\",\"toolkit\":\"CUDA 13.0.2\","
      "\"pinned_image\":\"qw38-cuda:13.0.2\",\"measurement_utc\":\"%s\","
      "\"production_shape\":{\"query_heads\":24,\"kv_heads\":4,\"head_width\":256,"
      "\"rotary_width\":64,\"chunk_rows\":64,\"kv_tile_rows\":32,\"threads\":256,"
      "\"group_size\":6},\"semantic\":{\"physical_index_bijective\":true,"
      "\"physical_index_bijective_capacity_3\":true,\"logical_pack_round_trip\":true,"
      "\"full_committed_tiles_coalesced\":true,"
      "\"production_one_row_output_exact\":true,\"candidate_bf16_exact\":true,"
      "\"finite_output\":true,\"prepare_isolation\":true,"
      "\"score_scratch_unchanged\":true,\"invalid_input_rejected\":true,"
      "\"commit_scatter_pack_equals_logical_prefix\":true,"
      "\"frontier_advances_once\":true},\"capacity\":{\"attention_cache_values\":%zu,"
      "\"sixteen_layer_kv_bytes\":%zu},\"round_trip_prefixes\":[1,2,31,32,33,65],"
      "\"tile_span_prefixes\":[0,31,32,2048],\"exact_row_counts\":[1,2,3,31,32,33,64,65],"
      "\"exact_prefixes\":[0,31,32],\"launch\":{\"kernel_nodes\":%d,"
      "\"staging_grid\":[%u,%u,%u],\"attention_grid\":[%u,%u,%u],"
      "\"staging_block\":[%u,%u,%u],\"attention_block\":[%u,%u,%u],"
      "\"dynamic_shared_bytes\":%u},\"proof_limit\":\"executed pointer-span / "
      "exact-value evidence, not Nsight DRAM transactions, latency, or "
      "end-to-end recovery\"}\n",
      prop.name, prop.major, prop.minor, driver / 1000, (driver % 1000) / 10,
      runtime / 1000, (runtime % 1000) / 10, utc, cache_values,
      sixteen_layer_bytes, nodes, staging_grid.x, staging_grid.y,
      staging_grid.z, attention_grid.x, attention_grid.y, attention_grid.z,
      staging_block.x, staging_block.y, staging_block.z, attention_block.x,
      attention_block.y, attention_block.z, dynamic_shared);
  return 0;
}
