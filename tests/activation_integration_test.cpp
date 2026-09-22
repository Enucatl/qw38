#include "activation_support.hpp"
#include "cuda/activation.hpp"
#include "cuda/buffer.hpp"
#include "cuda/stream.hpp"
#include "format/floatcvt.hpp"
#include "reference/math.hpp"

#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

using qw38::activation::test::bf16;
using qw38::activation::test::download_vec;
using qw38::activation::test::expect;
using qw38::activation::test::fail;
using qw38::activation::test::g_failures;
using qw38::activation::test::max_abs_diff_bf16;
using qw38::activation::test::upload_vec;
using qw38::cuda::DeviceBuffer;
using qw38::cuda::Stream;
using qw38::reference::kDefaultRmsEps;
using qw38::reference::kHeadDim;
using qw38::reference::kHidden;
namespace tol = qw38::reference::tol;

namespace {

constexpr std::uint32_t kQueryHeads = 24;

std::vector<std::uint16_t> ramp_h(std::uint32_t n, float scale, float bias) {
  std::vector<std::uint16_t> h(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    h[i] = bf16(bias + scale * (static_cast<float>(static_cast<int>(i % 13) - 6) /
                                6.0f));
  }
  return h;
}

}  // namespace

int main() {
  auto stream = Stream::create();
  expect(static_cast<bool>(stream), "create stream");
  if (!stream) {
    std::cerr << g_failures << " failures\n";
    return 1;
  }

  // Embedding → hidden RMS on a deterministic vocabulary row.
  std::uint32_t const vocab = 5;
  std::uint32_t const token = 3;
  auto table = ramp_h(vocab * kHidden, 1.8f, 0.2f);
  auto gamma = ramp_h(kHidden, 0.35f, 0.0f);
  std::vector<float> residual_cpu(kHidden);
  std::vector<std::uint16_t> norm_cpu(kHidden);
  expect(static_cast<bool>(qw38::reference::embed_gather_bf16_to_fp32(
             table, vocab, token, residual_cpu)),
         "cpu embed pipeline");
  expect(static_cast<bool>(qw38::reference::hidden_rms_norm_1p_gamma(
             residual_cpu, gamma, kDefaultRmsEps, norm_cpu)),
         "cpu hidden rms pipeline");

  auto d_table = upload_vec(table, *stream);
  auto d_gamma = upload_vec(gamma, *stream);
  auto d_residual = DeviceBuffer::allocate(kHidden * sizeof(float));
  auto d_norm = DeviceBuffer::allocate(kHidden * sizeof(std::uint16_t));
  expect(static_cast<bool>(d_table) && static_cast<bool>(d_gamma) &&
             static_cast<bool>(d_residual) && static_cast<bool>(d_norm),
         "pipeline alloc");
  if (!d_table || !d_gamma || !d_residual || !d_norm) {
    std::cerr << g_failures << " failures\n";
    return 1;
  }
  expect(static_cast<bool>(qw38::cuda::launch_embed_gather(
             static_cast<std::uint16_t const*>(d_table->data()), vocab, token,
             static_cast<float*>(d_residual->data()), *stream)),
         "launch embed pipeline");
  expect(static_cast<bool>(qw38::cuda::launch_hidden_rms(
             static_cast<float const*>(d_residual->data()),
             static_cast<std::uint16_t const*>(d_gamma->data()), kDefaultRmsEps,
             1, static_cast<std::uint16_t*>(d_norm->data()), *stream)),
         "launch hidden rms pipeline");
  auto got_res = download_vec<float>(*d_residual, kHidden, *stream);
  auto got_norm = download_vec<std::uint16_t>(*d_norm, kHidden, *stream);
  expect(static_cast<bool>(got_res) && static_cast<bool>(got_norm),
         "download embed-rms");
  if (got_res && got_norm) {
    expect(*got_res == residual_cpu, "pipeline residual exact");
    float d = max_abs_diff_bf16(*got_norm, norm_cpu);
    expect(d <= tol::kRmsBf16Abs, "pipeline norm within RMS tolerance");
  }

  // BF16 projection staging → fused FP32 QK RMS + RoPE → one BF16 store.
  // Distinct heads must not mix.
  std::vector<std::uint16_t> q(kQueryHeads * kHeadDim);
  for (std::uint32_t h = 0; h < kQueryHeads; ++h) {
    for (std::uint32_t i = 0; i < kHeadDim; ++i) {
      float const magnitude = 0.25f * static_cast<float>((h % 4) + 1);
      float const sign = ((i + h) % 3 == 0) ? -1.0f : 1.0f;
      q[h * kHeadDim + i] = bf16(sign * magnitude);
    }
  }
  auto q_gamma = ramp_h(kHeadDim, 0.2f, 0.0f);
  auto inv = qw38::reference::rope_inv_freq();
  std::int32_t const pos = 4096;
  std::vector<std::uint16_t> q_rope_cpu(kQueryHeads * kHeadDim);
  std::vector<std::uint16_t> q_double_rounded(kQueryHeads * kHeadDim);
  for (std::uint32_t h = 0; h < kQueryHeads; ++h) {
    auto in =
        std::span<std::uint16_t const>(q.data() + h * kHeadDim, kHeadDim);
    auto rot = std::span<std::uint16_t>(q_rope_cpu.data() + h * kHeadDim, kHeadDim);
    expect(static_cast<bool>(qw38::reference::qk_rms_rope_1p_gamma(
               in, q_gamma, kDefaultRmsEps, inv, pos, rot)),
           "cpu fused per-head qk-rope");

    std::vector<float> widened(kHeadDim);
    for (std::uint32_t i = 0; i < kHeadDim; ++i) {
      widened[i] = qw38::format::bf16_to_fp32(in[i]);
    }
    std::vector<std::uint16_t> rounded_norm(kHeadDim);
    auto rounded_rot = std::span<std::uint16_t>(
        q_double_rounded.data() + h * kHeadDim, kHeadDim);
    expect(static_cast<bool>(qw38::reference::qk_rms_norm_1p_gamma(
               widened, q_gamma, kDefaultRmsEps, rounded_norm)),
           "diagnostic rounded per-head qk");
    expect(static_cast<bool>(
               qw38::reference::partial_rope(rounded_norm, inv, pos, rounded_rot)),
           "diagnostic rounded per-head rope");
  }
  expect(q_rope_cpu != q_double_rounded,
         "single post-RoPE store differs from two rounded stores");

  auto d_q = upload_vec(q, *stream);
  auto d_qg = upload_vec(q_gamma, *stream);
  auto d_inv = upload_vec(std::vector<float>(inv.begin(), inv.end()), *stream);
  auto d_qr = DeviceBuffer::allocate(q_rope_cpu.size() * sizeof(std::uint16_t));
  expect(static_cast<bool>(d_q) && static_cast<bool>(d_qg) &&
             static_cast<bool>(d_inv) && static_cast<bool>(d_qr),
         "qk-rope alloc");
  if (!d_q || !d_qg || !d_inv || !d_qr) {
    std::cerr << g_failures << " failures\n";
    return 1;
  }
  expect(static_cast<bool>(qw38::cuda::launch_qk_rms_rope(
             static_cast<std::uint16_t const*>(d_q->data()),
             static_cast<std::uint16_t const*>(d_qg->data()), kDefaultRmsEps,
             static_cast<float const*>(d_inv->data()), pos, kQueryHeads,
             static_cast<std::uint16_t*>(d_qr->data()), *stream)),
         "launch fused qk-rope heads");
  auto got_qr = download_vec<std::uint16_t>(*d_qr, q_rope_cpu.size(), *stream);
  expect(static_cast<bool>(got_qr), "download qk-rope");
  if (got_qr) {
    float dr = max_abs_diff_bf16(*got_qr, q_rope_cpu);
    expect(dr <= tol::kRopeSmallAbs, "fused per-head QK/RoPE matches CPU");
    expect(*got_qr == q_rope_cpu,
           "CUDA preserves the single-store result bit-for-bit");
    expect(*got_qr != q_double_rounded,
           "CUDA does not reproduce the two-store pipeline");
  }

  if (g_failures != 0) {
    std::cerr << g_failures << " failures\n";
    return 1;
  }
  return 0;
}
