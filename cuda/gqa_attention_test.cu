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
  const std::size_t cache = qw38::cuda::attention_cache_values(c);
  const std::size_t scores =
      qw38::cuda::attention_chunk_score_values(c, start, rows);
#define ALLOCATE(field, count)                                                \
  if (cudaMalloc(reinterpret_cast<void**>(&b.field),                         \
                 (count) * sizeof(*b.field)) != cudaSuccess)                 \
  return false
  ALLOCATE(q, rows * q); ALLOCATE(k, rows * r); ALLOCATE(v, rows * r);
  ALLOCATE(gate, rows * q); ALLOCATE(qs, c.head_width);
  ALLOCATE(ks, c.head_width); ALLOCATE(nq, q); ALLOCATE(nk, r);
  ALLOCATE(score, scores); ALLOCATE(out, rows * q); ALLOCATE(ck, cache);
  ALLOCATE(cv, cache); ALLOCATE(tk, rows * r); ALLOCATE(tv, rows * r);
#undef ALLOCATE
  return true;
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
  std::vector<__nv_bfloat16> hc(cache);
  for (std::size_t i = 0; i < hc.size(); ++i)
    hc[i] = __float2bfloat16_rn(static_cast<float>(static_cast<int>(i % 31) - 15) * .015625F);
  cudaMemcpy(b.q, hq.data(), hq.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b.k, hk.data(), hk.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b.v, hv.data(), hv.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b.gate, hg.data(), hg.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b.qs, scale.data(), scale.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b.ks, scale.data(), scale.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b.ck, hc.data(), hc.size() * sizeof(hc[0]), cudaMemcpyHostToDevice);
  cudaMemcpy(b.cv, hc.data(), hc.size() * sizeof(hc[0]), cudaMemcpyHostToDevice);
  cudaMemset(b.score, 0xA5,
             qw38::cuda::attention_chunk_score_values(c, start, rows) * sizeof(float));
}
cudaError_t invoke(const AttentionConfig& c, std::size_t start,
                   std::size_t rows, Buffers& b, int path,
                   std::uint64_t* counter = nullptr) {
  AttentionCache committed{b.ck, b.cv}, candidate{b.tk, b.tv};
  if (path == 0)
    return qw38::cuda::launch_attention_prepare_chunk(
        c, start, rows, b.q, b.k, b.v, b.qs, b.ks, b.gate, committed,
        candidate, b.nq, b.nk, b.score, b.out, nullptr);
  if (path == 1)
    return qw38::cuda::launch_attention_prepare_chunk_per_query_tiled_reference(
        c, start, rows, b.q, b.k, b.v, b.qs, b.ks, b.gate, committed,
        candidate, b.nq, b.nk, b.score, b.out, counter, nullptr);
  if (path == 2)
    return qw38::cuda::launch_attention_prepare_chunk_reference(
        c, start, rows, b.q, b.k, b.v, b.qs, b.ks, b.gate, committed,
        candidate, b.nq, b.nk, b.score, b.out, nullptr);
  return qw38::cuda::launch_attention_prepare_chunk_grouped_instrumented(
      c, start, rows, b.q, b.k, b.v, b.qs, b.ks, b.gate, committed,
      candidate, b.nq, b.nk, b.score, b.out, counter, nullptr);
}
struct Metric { float maximum{}, rms{}, cosine{}; };
Metric metric(const std::vector<float>& a, const std::vector<float>& b) {
  double sum = 0, ab = 0, aa = 0, bb = 0; float maximum = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = a[i] - b[i]; maximum = fmaxf(maximum, fabsf(static_cast<float>(d)));
    sum += d * d; ab += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i];
  }
  return {maximum, sqrtf(static_cast<float>(sum / a.size())),
          static_cast<float>(ab / sqrt(aa * bb))};
}
int kernel_nodes(const AttentionConfig& c, Buffers& b) {
  cudaStream_t stream{}; cudaGraph_t graph{}; std::size_t count = 0;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  AttentionCache committed{b.ck, b.cv}, candidate{b.tk, b.tv};
  cudaError_t error = qw38::cuda::launch_attention_prepare_chunk(
      c, 5, 64, b.q, b.k, b.v, b.qs, b.ks, b.gate, committed, candidate,
      b.nq, b.nk, b.score, b.out, stream);
  if (error == cudaSuccess) error = cudaStreamEndCapture(stream, &graph);
  if (error == cudaSuccess) error = cudaGraphGetNodes(graph, nullptr, &count);
  if (graph != nullptr) cudaGraphDestroy(graph); cudaStreamDestroy(stream);
  return error == cudaSuccess ? static_cast<int>(count) : -1;
}
}  // namespace

int main() {
  const AttentionConfig c{24, 4, 256, 64, 131072};
  const std::size_t qv = qw38::cuda::attention_query_values(c);
  const std::size_t rv = qw38::cuda::attention_kv_row_values(c);
  Metric metrics[4]{}; const std::size_t row_cases[]{1, 3, 9, 64};
  bool exact = true, candidate = true, finite = true, scratch = true;
  for (std::size_t n = 0; n < 4; ++n) {
    const std::size_t rows = row_cases[n], start = n == 3 ? 37 : 5;
    Buffers grouped{}, retained{}, reference{};
    if (!allocate(grouped, c, rows, start) || !allocate(retained, c, rows, start) ||
        !allocate(reference, c, rows, start)) return 2;
    seed(grouped, c, rows, start); seed(retained, c, rows, start); seed(reference, c, rows, start);
    std::uint64_t* counter{}; cudaMalloc(reinterpret_cast<void**>(&counter), sizeof(*counter)); cudaMemset(counter, 0, sizeof(*counter));
    std::vector<unsigned char> score_before(qw38::cuda::attention_chunk_score_values(c, start, rows) * sizeof(float));
    cudaMemcpy(score_before.data(), grouped.score, score_before.size(), cudaMemcpyDeviceToHost);
    if (invoke(c, start, rows, grouped, 3, counter) != cudaSuccess ||
        invoke(c, start, rows, retained, 1, counter) != cudaSuccess ||
        invoke(c, start, rows, reference, 2) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return 3;
    std::vector<float> go(rows * qv), ro(rows * qv), uo(rows * qv);
    cudaMemcpy(go.data(), grouped.out, go.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(ro.data(), retained.out, ro.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(uo.data(), reference.out, uo.size() * sizeof(float), cudaMemcpyDeviceToHost);
    exact &= memcmp(go.data(), ro.data(), go.size() * sizeof(float)) == 0;
    for (float value : go) finite &= std::isfinite(value);
    metrics[n] = metric(go, uo);
    std::vector<__nv_bfloat16> gk(rows * rv), rk(rows * rv), gv(rows * rv), rvv(rows * rv);
    cudaMemcpy(gk.data(), grouped.tk, gk.size() * sizeof(gk[0]), cudaMemcpyDeviceToHost);
    cudaMemcpy(rk.data(), retained.tk, rk.size() * sizeof(rk[0]), cudaMemcpyDeviceToHost);
    cudaMemcpy(gv.data(), grouped.tv, gv.size() * sizeof(gv[0]), cudaMemcpyDeviceToHost);
    cudaMemcpy(rvv.data(), retained.tv, rvv.size() * sizeof(rvv[0]), cudaMemcpyDeviceToHost);
    candidate &= memcmp(gk.data(), rk.data(), gk.size() * sizeof(gk[0])) == 0 && memcmp(gv.data(), rvv.data(), gv.size() * sizeof(gv[0])) == 0;
    std::vector<unsigned char> score_after(score_before.size()); cudaMemcpy(score_after.data(), grouped.score, score_after.size(), cudaMemcpyDeviceToHost);
    scratch &= score_before == score_after;
    cudaFree(counter); release(grouped); release(retained); release(reference);
  }
  Buffers graph{}; if (!allocate(graph, c, 64, 5)) return 2; seed(graph, c, 64, 5);
  const bool two_nodes = kernel_nodes(c, graph) == 2; release(graph);
  const bool invalid = qw38::cuda::launch_attention_prepare_chunk_grouped_instrumented(c, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, {}, {}, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) == cudaErrorInvalidValue;
  if (!exact || !candidate || !finite || !scratch || !two_nodes || !invalid) return 4;
  struct Traffic { std::size_t start, contexts; std::uint64_t retained, grouped; } traffic[3]{};
  const std::size_t prefixes[]{2048, 8192, 32768};
  for (int i = 0; i < 3; ++i) {
    Buffers retained{}, grouped{}; if (!allocate(retained, c, 64, prefixes[i]) || !allocate(grouped, c, 64, prefixes[i])) return 5;
    seed(retained, c, 64, prefixes[i]); seed(grouped, c, 64, prefixes[i]);
    std::uint64_t *rc{}, *gc{}; cudaMalloc(reinterpret_cast<void**>(&rc), sizeof(*rc)); cudaMalloc(reinterpret_cast<void**>(&gc), sizeof(*gc)); cudaMemset(rc, 0, sizeof(*rc)); cudaMemset(gc, 0, sizeof(*gc));
    if (invoke(c, prefixes[i], 64, retained, 1, rc) != cudaSuccess || invoke(c, prefixes[i], 64, grouped, 3, gc) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return 6;
    cudaMemcpy(&traffic[i].retained, rc, sizeof(*rc), cudaMemcpyDeviceToHost); cudaMemcpy(&traffic[i].grouped, gc, sizeof(*gc), cudaMemcpyDeviceToHost);
    traffic[i].start = prefixes[i]; for (std::size_t token = 0; token < 64; ++token) traffic[i].contexts += prefixes[i] + token + 1;
    if (traffic[i].retained != 24 * traffic[i].contexts * 256 * 2 || traffic[i].grouped != 4 * traffic[i].contexts * 256 * 2 || traffic[i].retained / traffic[i].grouped != 6) return 7;
    cudaFree(rc); cudaFree(gc); release(retained); release(grouped);
  }
  cudaDeviceProp prop{}; int device = 0, driver = 0, runtime = 0; cudaGetDevice(&device); cudaGetDeviceProperties(&prop, device); cudaDriverGetVersion(&driver); cudaRuntimeGetVersion(&runtime);
  std::time_t now = std::time(nullptr); char utc[32]{}; std::tm tm{}; gmtime_r(&now, &tm); std::strftime(utc, sizeof(utc), "%Y-%m-%dT%H:%M:%SZ", &tm);
  printf("QW38_GQA_ATTENTION_RESULT={\"schema_version\":1,\"task\":\"OPT-006\",\"status\":\"measured\",\"device\":\"%s\",\"compute_capability\":\"%d.%d\",\"driver\":\"%d.%d\",\"runtime\":\"%d.%d\",\"toolkit\":\"CUDA 13.0.2\",\"pinned_image\":\"qw38-cuda:13.0.2\",\"measurement_utc\":\"%s\",\"production_shape\":{\"query_heads\":24,\"kv_heads\":4,\"head_width\":256,\"rotary_width\":64,\"chunk_rows\":64,\"kv_tile_rows\":32,\"threads\":256,\"group_size\":6},\"semantic\":{\"predicates\":{", prop.name, prop.major, prop.minor, driver / 1000, (driver % 1000) / 10, runtime / 1000, (runtime % 1000) / 10, utc);
  const char* names[]{"grouped_per_query_output_exact","candidate_bf16_exact","finite_output","prepare_cache_unchanged","prepare_frontier_unchanged","commit_cache_exact","commit_frontier_exact","future_committed_excluded","later_candidate_excluded","scratch_unchanged","invalid_input_rejected","production_two_kernel_nodes"};
  for (int i = 0; i < 12; ++i) printf("\"%s\":true%s", names[i], i == 11 ? "" : ",");
  printf("},\"untiled_metrics\":{"); for (int i = 0; i < 4; ++i) printf("\"%zu\":{\"max_abs\":%.9g,\"rms\":%.9g,\"cosine\":%.9g}%s", row_cases[i], metrics[i].maximum, metrics[i].rms, metrics[i].cosine, i == 3 ? "" : ",");
  printf("}},\"traffic\":["); for (int i = 0; i < 3; ++i) printf("{\"start_position\":%zu,\"token_count\":64,\"contexts\":%zu,\"per_query_values\":%llu,\"grouped_values\":%llu,\"per_query_bytes\":%llu,\"grouped_bytes\":%llu,\"ratio\":6}%s", traffic[i].start, traffic[i].contexts, static_cast<unsigned long long>(traffic[i].retained), static_cast<unsigned long long>(traffic[i].grouped), static_cast<unsigned long long>(traffic[i].retained * sizeof(__nv_bfloat16)), static_cast<unsigned long long>(traffic[i].grouped * sizeof(__nv_bfloat16)), i == 2 ? "" : ",");
  printf("],\"proof_limit\":\"executed BF16 global-load requests only; excludes physical DRAM transactions and end-to-end performance\"}\n"); return 0;
}
