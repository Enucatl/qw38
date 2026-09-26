#include "cuda/attention.hpp"
#include "cuda/buffer.hpp"
#include "cuda/copy.hpp"
#include "cuda/stream.hpp"
#include "format/floatcvt.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

namespace {
constexpr std::uint32_t H = qw38::cuda::kAttnQueryHeads;
constexpr std::uint32_t KVH = qw38::cuda::kAttnKvHeads;
constexpr std::uint32_t D = qw38::cuda::kAttnHeadDim;
constexpr std::uint32_t C = 337;
constexpr std::uint32_t M = 65;
using qw38::format::bf16_to_fp32;
using qw38::format::fp32_to_bf16_rne;

std::size_t kv_index(std::uint32_t component, std::uint32_t head,
                     std::uint32_t token, std::uint32_t d) {
  return ((component * KVH + head) * C + token) * D + d;
}

template <typename T>
bool upload(qw38::cuda::DeviceBuffer& dst, std::vector<T> const& src,
            qw38::cuda::Stream const& stream) {
  return bool(qw38::cuda::copy_h2d(dst.data(), std::as_bytes(std::span(src)), stream));
}

template <typename T>
std::vector<T> download(qw38::cuda::DeviceBuffer const& src, std::size_t n,
                        qw38::cuda::Stream const& stream) {
  std::vector<T> out(n);
  if (!qw38::cuda::copy_d2h(out.data(), src.data(), n * sizeof(T), stream) ||
      !stream.sync()) return {};
  return out;
}

bool scan_case(qw38::cuda::Stream const& stream, std::uint32_t first,
               std::uint32_t count, std::uint32_t query_tile) {
  std::vector<std::uint16_t> q((M + 1u) * H * D), g(q.size());
  std::vector<std::uint16_t> kv(16u * 2u * KVH * C * D,
                               fp32_to_bf16_rne(777.0f));
  for (std::uint32_t t = 0; t < count; ++t)
    for (std::uint32_t h = 0; h < H; ++h) {
      std::size_t const off = (t * H + h) * D;
      // Scores start above FP32 exp overflow (about 88), then a larger
      // maximum arrives beyond the 64-key tile boundary. Earlier values
      // still contribute materially to the normalized result.
      q[off] = fp32_to_bf16_rne(256.0f);
      q[off + 1u] = fp32_to_bf16_rne(0.25f * (t + 1u));
      for (std::uint32_t d = 2; d < D; ++d)
        q[off + d] = fp32_to_bf16_rne(
            0.01f * (static_cast<int>((d * 7u + h * 3u + t) % 23u) - 11));
      for (std::uint32_t d = 0; d < D; ++d)
        g[off + d] = fp32_to_bf16_rne((d % 3u == 0u ? -1.0f : 1.0f) *
                                       (h == 23u ? 80.0f : 0.6f));
    }
  for (std::uint32_t head = 0; head < KVH; ++head)
    for (std::uint32_t t = 0; t < first + count; ++t) {
      kv[kv_index(0, head, t, 0)] = fp32_to_bf16_rne(
          (t < 64u ? 6.25f : 6.4375f) + 0.0625f * head);
      kv[kv_index(0, head, t, 1)] = fp32_to_bf16_rne(0.2f * t);
      for (std::uint32_t d = 2; d < D; ++d)
        kv[kv_index(0, head, t, d)] = fp32_to_bf16_rne(
            0.02f * (static_cast<int>((d * 3u + head + t * 5u) % 29u) - 14));
      for (std::uint32_t d = 0; d < D; ++d)
        kv[kv_index(1, head, t, d)] = fp32_to_bf16_rne(
            (t < 64u ? 0.05f : 0.8f) * static_cast<float>(head + 1u) +
            0.001f * d);
    }
  auto dq = qw38::cuda::DeviceBuffer::allocate(q.size() * 2u);
  auto dg = qw38::cuda::DeviceBuffer::allocate(g.size() * 2u);
  auto dkv = qw38::cuda::DeviceBuffer::allocate(kv.size() * 2u);
  auto dy = qw38::cuda::DeviceBuffer::allocate(q.size() * 2u);
  if (!dq || !dg || !dkv || !dy || !upload(*dq, q, stream) ||
      !upload(*dg, g, stream) || !upload(*dkv, kv, stream)) return false;
  std::vector<std::uint16_t> guard(q.size(), fp32_to_bf16_rne(-777.0f));
  if (!upload(*dy, guard, stream)) return false;
  if (qw38::cuda::launch_attention_prefill_scan(
          q.data(), static_cast<std::uint16_t const*>(dg->data()),
          static_cast<std::uint16_t const*>(dkv->data()), 0, C, first, count,
          static_cast<std::uint16_t*>(dy->data()), stream, query_tile))
    return false;
  auto st = qw38::cuda::launch_attention_prefill_scan(
      static_cast<std::uint16_t const*>(dq->data()),
      static_cast<std::uint16_t const*>(dg->data()),
      static_cast<std::uint16_t const*>(dkv->data()), 0, C, first, count,
      static_cast<std::uint16_t*>(dy->data()), stream, query_tile);
  if (!st) { std::cerr << qw38::cuda::error_message(st.error()) << '\n'; return false; }
  auto got = download<std::uint16_t>(*dy, q.size(), stream);
  if (got.size() != q.size()) return false;
  if (!std::equal(got.begin() + count * H * D, got.end(),
                  guard.begin() + count * H * D)) return false;
  // Reject capacity overflow before touching output, then prove exact replay.
  if (qw38::cuda::launch_attention_prefill_scan(
          static_cast<std::uint16_t const*>(dq->data()),
          static_cast<std::uint16_t const*>(dg->data()),
          static_cast<std::uint16_t const*>(dkv->data()), 0, C, C - count + 1u,
          count, static_cast<std::uint16_t*>(dy->data()), stream, query_tile) ||
      got != download<std::uint16_t>(*dy, q.size(), stream)) return false;
  if (!qw38::cuda::launch_attention_prefill_scan(
          static_cast<std::uint16_t const*>(dq->data()),
          static_cast<std::uint16_t const*>(dg->data()),
          static_cast<std::uint16_t const*>(dkv->data()), 0, C, first, count,
          static_cast<std::uint16_t*>(dy->data()), stream, query_tile) ||
      got != download<std::uint16_t>(*dy, q.size(), stream)) return false;
  // The last row also goes through the production 256-key segmented scan and
  // merge. Dense asymmetric Q/K exercise every lane of the new warp reduction.
  std::uint32_t const segments = (first + count + 255u) / 256u;
  auto partials = qw38::cuda::DeviceBuffer::allocate(
      H * segments * qw38::cuda::kAttnPartialStride * sizeof(float));
  std::size_t const last_off = (count - 1u) * H * D;
  if (!partials || !qw38::cuda::launch_attention_scan(
          static_cast<std::uint16_t const*>(dq->data()) + last_off,
          static_cast<std::uint16_t const*>(dkv->data()), 0, C, first + count,
          static_cast<float*>(partials->data()), segments, stream) ||
      !qw38::cuda::launch_attention_merge(
          static_cast<float const*>(partials->data()),
          static_cast<std::uint16_t const*>(dg->data()) + last_off, segments,
          static_cast<std::uint16_t*>(dy->data()) + last_off, stream)) return false;
  auto decoded = download<std::uint16_t>(*dy, q.size(), stream);
  if (decoded.size() != q.size()) return false;
  float max_diff = 0.0f;
  for (std::uint32_t t = 0; t < count; ++t)
    for (std::uint32_t h = 0; h < H; ++h) {
      auto const head = h / qw38::cuda::kAttnGqaGroup;
      std::size_t const off = (t * H + h) * D;
      std::vector<double> scores(first + t + 1u);
      double best = -INFINITY;
      for (std::uint32_t key = 0; key <= first + t; ++key) {
        double dot = 0.0;
        for (std::uint32_t d = 0; d < D; ++d)
          dot += static_cast<double>(bf16_to_fp32(q[off + d])) *
                 bf16_to_fp32(kv[kv_index(0, head, key, d)]);
        scores[key] = dot * qw38::cuda::kAttnScale;
        best = std::max(best, scores[key]);
      }
      double sum = 0.0;
      for (double& score : scores) { score = std::exp(score - best); sum += score; }
      for (std::uint32_t d = 0; d < D; ++d) {
        double acc = 0.0;
        for (std::uint32_t key = 0; key < scores.size(); ++key)
          acc += scores[key] * bf16_to_fp32(kv[kv_index(1, head, key, d)]);
        double const gate = 1.0 / (1.0 + std::exp(-bf16_to_fp32(g[off + d])));
        float const expected = bf16_to_fp32(fp32_to_bf16_rne(
            static_cast<float>(acc / sum * gate)));
        float const value = bf16_to_fp32(got[off + d]);
        max_diff = std::max(max_diff, std::fabs(value - expected));
        if (!std::isfinite(value) || std::fabs(value - expected) > 0.016f) {
          std::cerr << "scan mismatch first=" << first << " row=" << t
                    << " head=" << h << " dim=" << d << " got=" << value
                    << " expected=" << expected << '\n';
          return false;
        }
        if (t + 1u == count) {
          float const decode_value = bf16_to_fp32(decoded[off + d]);
          if (!std::isfinite(decode_value) ||
              std::fabs(decode_value - expected) > 0.016f) {
            std::cerr << "segmented decode mismatch\n";
            return false;
          }
        }
      }
    }
  std::cout << "scan first=" << first << " count=" << count
            << " query_tile=" << query_tile
            << " max_bf16_diff=" << max_diff << '\n';
  return true;
}
}  // namespace

int main() {
  auto stream = qw38::cuda::Stream::create();
  if (!stream) return 1;
  auto resources = qw38::cuda::attention_prefill_resources();
  if (!resources || resources->local_bytes != 0 ||
      resources->occupancy_blocks_per_sm == 0) return 1;
  std::cout << "mma query_tile=" << qw38::cuda::kAttnPrefillQueryTile
            << " key_tile=" << qw38::cuda::kAttnPrefillKeyTile
            << " registers=" << resources->registers
            << " shared_bytes=" << resources->shared_bytes
            << " local_bytes=" << resources->local_bytes
            << " blocks_per_sm=" << resources->occupancy_blocks_per_sm << '\n';
  for (std::uint32_t tile : {1u, 4u, qw38::cuda::kAttnPrefillQueryTile})
    if (!scan_case(*stream, 0, 3, tile) ||
        !scan_case(*stream, 33, 7, tile)) return 1;
  for (std::uint32_t count : {31u, 32u, 33u, 63u, 64u, 65u})
    if (!scan_case(*stream, 0, count, qw38::cuda::kAttnPrefillQueryTile)) return 1;
  if (!scan_case(*stream, 63, 2, qw38::cuda::kAttnPrefillQueryTile) ||
      !scan_case(*stream, 271, 65, qw38::cuda::kAttnPrefillQueryTile)) return 1;
  return 0;
}
