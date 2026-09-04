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
struct Buffers {
  float *q{}, *k{}, *v{}, *gate{}, *qs{}, *ks{}, *nq{}, *nk{}, *score{},
      *out{};
  __nv_bfloat16 *ck{}, *cv{}, *tk{}, *tv{};
  std::size_t cache_values{};
};
void release(Buffers& b) {
  cudaFree(b.tv); cudaFree(b.tk); cudaFree(b.cv); cudaFree(b.ck);
  cudaFree(b.out); cudaFree(b.score); cudaFree(b.nk); cudaFree(b.nq);
  cudaFree(b.ks); cudaFree(b.qs); cudaFree(b.gate); cudaFree(b.v);
  cudaFree(b.k); cudaFree(b.q);
}
bool allocate(Buffers& b, const AttentionConfig& c, std::size_t rows,
              std::size_t start) {
  const std::size_t q = qw38::cuda::attention_query_values(c);
  const std::size_t r = qw38::cuda::attention_kv_row_values(c);
  b.cache_values = (start + rows + 1) * r;
  const std::size_t scores =
      qw38::cuda::attention_chunk_score_values(c, start, rows);
#define ALLOCATE(field, count)                                             \
  if (cudaMalloc(reinterpret_cast<void**>(&b.field),                       \
                 (count) * sizeof(*b.field)) != cudaSuccess)               \
  return false
  ALLOCATE(q, rows * q); ALLOCATE(k, rows * r); ALLOCATE(v, rows * r);
  ALLOCATE(gate, rows * q); ALLOCATE(qs, c.head_width);
  ALLOCATE(ks, c.head_width); ALLOCATE(nq, q); ALLOCATE(nk, r);
  ALLOCATE(score, scores); ALLOCATE(out, rows * q);
  ALLOCATE(ck, b.cache_values); ALLOCATE(cv, b.cache_values);
  ALLOCATE(tk, rows * r); ALLOCATE(tv, rows * r);
#undef ALLOCATE
  return true;
}
void seed(Buffers& b, const AttentionConfig& c, std::size_t rows,
          std::size_t start) {
  const std::size_t q = qw38::cuda::attention_query_values(c);
  const std::size_t r = qw38::cuda::attention_kv_row_values(c);
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
  std::vector<__nv_bfloat16> hc(b.cache_values);
  for (std::size_t i = 0; i < hc.size(); ++i)
    hc[i] = __float2bfloat16_rn(
        static_cast<float>(static_cast<int>(i % 31) - 15) * .015625F);
  cudaMemcpy(b.q, hq.data(), hq.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b.k, hk.data(), hk.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b.v, hv.data(), hv.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b.gate, hg.data(), hg.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b.qs, scale.data(), scale.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b.ks, scale.data(), scale.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b.ck, hc.data(), hc.size() * sizeof(hc[0]), cudaMemcpyHostToDevice);
  cudaMemcpy(b.cv, hc.data(), hc.size() * sizeof(hc[0]), cudaMemcpyHostToDevice);
  cudaMemset(b.score, 0xA5,
             qw38::cuda::attention_chunk_score_values(c, start, rows) *
                 sizeof(float));
}
cudaError_t prepare(const AttentionConfig& c, std::size_t start,
                    std::size_t rows, Buffers& b, bool production,
                    std::size_t input_offset = 0,
                    std::size_t candidate_offset = 0,
                    std::size_t output_offset = 0) {
  const std::size_t q = qw38::cuda::attention_query_values(c);
  const std::size_t r = qw38::cuda::attention_kv_row_values(c);
  AttentionCache committed{b.ck, b.cv};
  AttentionCache candidate{b.tk + candidate_offset * r,
                           b.tv + candidate_offset * r};
  if (production)
    return qw38::cuda::launch_attention_prepare_chunk(
        c, start, rows, b.q + input_offset * q, b.k + input_offset * r,
        b.v + input_offset * r, b.qs, b.ks, b.gate + input_offset * q,
        committed, candidate, b.nq, b.nk, b.score,
        b.out + output_offset * q, nullptr);
  std::uint64_t* count{};
  cudaMalloc(reinterpret_cast<void**>(&count), sizeof(*count));
  cudaMemset(count, 0, sizeof(*count));
  const cudaError_t error =
      qw38::cuda::launch_attention_prepare_chunk_grouped_instrumented(
          c, start, rows, b.q, b.k, b.v, b.qs, b.ks, b.gate, committed,
          candidate, b.nq, b.nk, b.score, b.out, count, nullptr);
  cudaFree(count);
  return error;
}
struct Launch {
  dim3 staging_grid{}, attention_grid{}, staging_block{}, attention_block{};
  unsigned attention_dynamic_shared{};
  int nodes{}, registers{}, static_shared{}, local_bytes{}, max_dynamic{},
      active_blocks{};
  void* function{};
};
bool inspect(const AttentionConfig& c, std::size_t rows, Buffers& b,
             Launch& result) {
  cudaStream_t stream{};
  cudaGraph_t graph{};
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
    return false;
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  AttentionCache committed{b.ck, b.cv}, candidate{b.tk, b.tv};
  cudaError_t error = qw38::cuda::launch_attention_prepare_chunk(
      c, 31, rows, b.q, b.k, b.v, b.qs, b.ks, b.gate, committed, candidate,
      b.nq, b.nk, b.score, b.out, stream);
  if (error == cudaSuccess) error = cudaStreamEndCapture(stream, &graph);
  std::size_t count = 0;
  if (error == cudaSuccess) error = cudaGraphGetNodes(graph, nullptr, &count);
  std::vector<cudaGraphNode_t> nodes(count);
  if (error == cudaSuccess) error = cudaGraphGetNodes(graph, nodes.data(), &count);
  result.nodes = static_cast<int>(count);
  for (cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type{};
    cudaKernelNodeParams params{};
    if (cudaGraphNodeGetType(node, &type) != cudaSuccess ||
        type != cudaGraphNodeTypeKernel ||
        cudaGraphKernelNodeGetParams(node, &params) != cudaSuccess) {
      error = cudaErrorInvalidValue;
      break;
    }
    if (params.sharedMemBytes == 0) {
      result.staging_grid = params.gridDim;
      result.staging_block = params.blockDim;
    } else {
      result.attention_grid = params.gridDim;
      result.attention_block = params.blockDim;
      result.attention_dynamic_shared = params.sharedMemBytes;
      result.function = params.func;
      if (params.sharedMemBytes != 33792)
        error = cudaErrorInvalidConfiguration;
    }
  }
  if (error == cudaSuccess && result.function != nullptr && rows == 64) {
    cudaFuncAttributes attr{};
    error = cudaFuncGetAttributes(&attr, result.function);
    result.registers = attr.numRegs;
    result.static_shared = static_cast<int>(attr.sharedSizeBytes);
    result.local_bytes = static_cast<int>(attr.localSizeBytes);
    result.max_dynamic = attr.maxDynamicSharedSizeBytes;
    if (error == cudaSuccess)
      error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &result.active_blocks, result.function, 256, 33792);
  }
  if (graph != nullptr) cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  return error == cudaSuccess && result.nodes == 2 &&
         result.staging_grid.x == c.kv_heads && result.staging_grid.y == rows &&
         result.staging_grid.z == 1 && result.attention_grid.x == c.kv_heads &&
         result.attention_grid.y == (rows + 1) / 2 &&
         result.attention_grid.z == 1 && result.staging_block.x == 256 &&
         result.staging_block.y == 1 && result.staging_block.z == 1 &&
         result.attention_block.x == 256 && result.attention_block.y == 1 &&
         result.attention_block.z == 1;
}
}  // namespace

int main() {
  const AttentionConfig c{24, 4, 256, 64, 131072};
  const std::size_t q = qw38::cuda::attention_query_values(c);
  const std::size_t r = qw38::cuda::attention_kv_row_values(c);
  const std::size_t cases[]{1, 2, 3, 63, 64, 65};
  Launch launches[6]{};
  bool exact = true, candidate = true, finite = true, cache_unchanged = true,
       frontier_unchanged = true, scratch = true, topology = true;
  for (std::size_t n = 0; n < 6; ++n) {
    Buffers production{}, reference{};
    if (!allocate(production, c, cases[n], 31) ||
        !allocate(reference, c, cases[n], 31))
      return 2;
    seed(production, c, cases[n], 31); seed(reference, c, cases[n], 31);
    std::vector<__nv_bfloat16> cache_before(production.cache_values * 2);
    cudaMemcpy(cache_before.data(), production.ck,
               production.cache_values * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(cache_before.data() + production.cache_values, production.cv,
               production.cache_values * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);
    std::vector<unsigned char> score_before(
        qw38::cuda::attention_chunk_score_values(c, 31, cases[n]) * sizeof(float));
    cudaMemcpy(score_before.data(), production.score, score_before.size(),
               cudaMemcpyDeviceToHost);
    std::uint64_t frontier_value = 31, *frontier{};
    cudaMalloc(reinterpret_cast<void**>(&frontier), sizeof(*frontier));
    cudaMemcpy(frontier, &frontier_value, sizeof(frontier_value), cudaMemcpyHostToDevice);
    if (prepare(c, 31, cases[n], production, true) != cudaSuccess ||
        prepare(c, 31, cases[n], reference, false) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess)
      return 3;
    std::vector<float> po(cases[n] * q), ro(cases[n] * q);
    cudaMemcpy(po.data(), production.out, po.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(ro.data(), reference.out, ro.size() * sizeof(float), cudaMemcpyDeviceToHost);
    exact &= memcmp(po.data(), ro.data(), po.size() * sizeof(float)) == 0;
    for (float value : po) finite &= std::isfinite(value);
    std::vector<__nv_bfloat16> pk(cases[n] * r), rk(cases[n] * r),
        pv(cases[n] * r), rv(cases[n] * r);
    cudaMemcpy(pk.data(), production.tk, pk.size() * sizeof(pk[0]), cudaMemcpyDeviceToHost);
    cudaMemcpy(rk.data(), reference.tk, rk.size() * sizeof(rk[0]), cudaMemcpyDeviceToHost);
    cudaMemcpy(pv.data(), production.tv, pv.size() * sizeof(pv[0]), cudaMemcpyDeviceToHost);
    cudaMemcpy(rv.data(), reference.tv, rv.size() * sizeof(rv[0]), cudaMemcpyDeviceToHost);
    candidate &= memcmp(pk.data(), rk.data(), pk.size() * sizeof(pk[0])) == 0 &&
                 memcmp(pv.data(), rv.data(), pv.size() * sizeof(pv[0])) == 0;
    std::vector<__nv_bfloat16> cache_after(cache_before.size());
    cudaMemcpy(cache_after.data(), production.ck,
               production.cache_values * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(cache_after.data() + production.cache_values, production.cv,
               production.cache_values * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cache_unchanged &= cache_before == cache_after;
    cudaMemcpy(&frontier_value, frontier, sizeof(frontier_value), cudaMemcpyDeviceToHost);
    frontier_unchanged &= frontier_value == 31;
    std::vector<unsigned char> score_after(score_before.size());
    cudaMemcpy(score_after.data(), production.score, score_after.size(), cudaMemcpyDeviceToHost);
    scratch &= score_before == score_after;
    topology &= inspect(c, cases[n], production, launches[n]);
    cudaFree(frontier); release(production); release(reference);
  }
  Buffers single{}, split{};
  if (!allocate(single, c, 65, 31) || !allocate(split, c, 65, 31)) return 2;
  seed(single, c, 65, 31); seed(split, c, 65, 31);
  std::uint64_t one_value = 31, split_value = 31, *one_frontier{}, *split_frontier{};
  cudaMalloc(reinterpret_cast<void**>(&one_frontier), sizeof(*one_frontier));
  cudaMalloc(reinterpret_cast<void**>(&split_frontier), sizeof(*split_frontier));
  cudaMemcpy(one_frontier, &one_value, sizeof(one_value), cudaMemcpyHostToDevice);
  cudaMemcpy(split_frontier, &split_value, sizeof(split_value), cudaMemcpyHostToDevice);
  bool boundary = prepare(c, 31, 65, single, true) == cudaSuccess &&
      qw38::cuda::launch_attention_commit_chunk(c, 31, 65, {single.tk, single.tv},
          {single.ck, single.cv}, 96, one_frontier, nullptr) == cudaSuccess &&
      prepare(c, 31, 64, split, true) == cudaSuccess &&
      qw38::cuda::launch_attention_commit_chunk(c, 31, 64, {split.tk, split.tv},
          {split.ck, split.cv}, 95, split_frontier, nullptr) == cudaSuccess &&
      cudaDeviceSynchronize() == cudaSuccess &&
      prepare(c, 95, 1, split, true, 64, 64, 64) == cudaSuccess &&
      qw38::cuda::launch_attention_commit_chunk(c, 95, 1,
          {split.tk + 64 * r, split.tv + 64 * r}, {split.ck, split.cv}, 96,
          split_frontier, nullptr) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
  std::vector<float> one_out(65 * q), split_out(65 * q);
  std::vector<__nv_bfloat16> one_cache(65 * r * 2), split_cache(65 * r * 2);
  cudaMemcpy(one_out.data(), single.out, one_out.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(split_out.data(), split.out, split_out.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(one_cache.data(), single.ck + 31 * r, 65 * r * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
  cudaMemcpy(one_cache.data() + 65 * r, single.cv + 31 * r, 65 * r * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
  cudaMemcpy(split_cache.data(), split.ck + 31 * r, 65 * r * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
  cudaMemcpy(split_cache.data() + 65 * r, split.cv + 31 * r, 65 * r * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
  cudaMemcpy(&one_value, one_frontier, sizeof(one_value), cudaMemcpyDeviceToHost);
  cudaMemcpy(&split_value, split_frontier, sizeof(split_value), cudaMemcpyDeviceToHost);
  boundary &= one_out == split_out && one_cache == split_cache &&
              one_value == 96 && split_value == 96;
  cudaFree(one_frontier); cudaFree(split_frontier); release(single); release(split);
  const bool invalid = qw38::cuda::launch_attention_prepare_chunk(
      c, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, {}, {},
      nullptr, nullptr, nullptr, nullptr, nullptr) == cudaErrorInvalidValue;
  const Launch& measured = launches[4];
  if (!exact || !candidate || !finite || !cache_unchanged ||
      !frontier_unchanged || !scratch || !topology || !boundary || !invalid ||
      measured.local_bytes > 512 || measured.active_blocks < 1 ||
      measured.registers < 1)
    return 4;
  cudaDeviceProp prop{}; int device = 0, driver = 0, runtime = 0;
  cudaGetDevice(&device); cudaGetDeviceProperties(&prop, device);
  cudaDriverGetVersion(&driver); cudaRuntimeGetVersion(&runtime);
  std::time_t now = std::time(nullptr); char utc[32]{}; std::tm tm{};
  gmtime_r(&now, &tm); std::strftime(utc, sizeof(utc), "%Y-%m-%dT%H:%M:%SZ", &tm);
  printf("QW38_QUERY_ROW_ATTENTION_RESULT={\"schema_version\":1,\"task\":\"OPT-007\",\"status\":\"measured\",\"device\":\"%s\",\"compute_capability\":\"%d.%d\",\"driver\":\"%d.%d\",\"runtime\":\"%d.%d\",\"toolkit\":\"CUDA 13.0.2\",\"pinned_image\":\"qw38-cuda:13.0.2\",\"measurement_utc\":\"%s\",\"production_shape\":{\"query_heads\":24,\"kv_heads\":4,\"head_width\":256,\"rotary_width\":64,\"chunk_rows\":64,\"kv_tile_rows\":32,\"query_rows_per_block\":2,\"threads\":256,\"group_size\":6},\"semantic\":{\"production_reference_output_exact\":true,\"candidate_bf16_exact\":true,\"finite_output\":true,\"prepare_cache_unchanged\":true,\"prepare_frontier_unchanged\":true,\"score_scratch_unchanged\":true,\"invalid_input_rejected\":true,\"chunk_65_equals_64_plus_1_output\":true,\"chunk_65_equals_64_plus_1_cache\":true,\"chunk_65_equals_64_plus_1_frontier\":true},\"launches\":[", prop.name, prop.major, prop.minor, driver / 1000, (driver % 1000) / 10, runtime / 1000, (runtime % 1000) / 10, utc);
  for (int i = 0; i < 6; ++i)
    printf("{\"rows\":%zu,\"kernel_nodes\":%d,\"staging_grid\":[%u,%u,%u],\"staging_block\":[%u,%u,%u],\"attention_grid\":[%u,%u,%u],\"attention_block\":[%u,%u,%u],\"dynamic_shared_bytes\":%u}%s", cases[i], launches[i].nodes, launches[i].staging_grid.x, launches[i].staging_grid.y, launches[i].staging_grid.z, launches[i].staging_block.x, launches[i].staging_block.y, launches[i].staging_block.z, launches[i].attention_grid.x, launches[i].attention_grid.y, launches[i].attention_grid.z, launches[i].attention_block.x, launches[i].attention_block.y, launches[i].attention_block.z, launches[i].attention_dynamic_shared, i == 5 ? "" : ",");
  printf("],\"kernel_attributes\":{\"registers\":%d,\"static_shared_bytes\":%d,\"local_bytes_per_thread\":%d,\"maximum_dynamic_shared_bytes\":%d,\"active_blocks_per_sm\":%d,\"sm_count\":%d,\"launch_blocks\":%u},\"proof_limit\":\"component-only exact semantic and launch evidence; no end-to-end performance or speedup claim\"}\n", measured.registers, measured.static_shared, measured.local_bytes, measured.max_dynamic, measured.active_blocks, prop.multiProcessorCount, measured.attention_grid.x * measured.attention_grid.y * measured.attention_grid.z);
  return 0;
}
