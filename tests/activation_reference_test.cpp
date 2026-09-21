#include "activation_support.hpp"
#include "cuda/activation.hpp"
#include "cuda/buffer.hpp"
#include "cuda/stream.hpp"
#include "format/floatcvt.hpp"
#include "reference/math.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

using qw38::activation::test::almost_equal;
using qw38::activation::test::bf16;
using qw38::activation::test::download_vec;
using qw38::activation::test::expect;
using qw38::activation::test::fail;
using qw38::activation::test::g_failures;
using qw38::activation::test::max_abs_diff_bf16;
using qw38::activation::test::upload_vec;
using qw38::cuda::DeviceBuffer;
using qw38::cuda::ErrorCode;
using qw38::cuda::Stream;
using qw38::reference::kDefaultRmsEps;
using qw38::reference::kGdnHeadDim;
using qw38::reference::kHeadDim;
using qw38::reference::kHidden;
namespace tol = qw38::reference::tol;

namespace {

std::vector<float> ramp_f(std::uint32_t n, float scale, float offset = 0.0f) {
  std::vector<float> out(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    out[i] = offset + scale * (static_cast<float>(static_cast<int>(i % 19) - 9) /
                               9.0f);
  }
  return out;
}

std::vector<std::uint16_t> ramp_h(std::uint32_t n, float scale) {
  auto f = ramp_f(n, scale);
  std::vector<std::uint16_t> h(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    h[i] = bf16(f[i]);
  }
  return h;
}

std::vector<std::uint16_t> filled_h(std::uint32_t n, float v) {
  return std::vector<std::uint16_t>(n, bf16(v));
}

void expect_bf16_close(std::span<std::uint16_t const> got,
                       std::span<std::uint16_t const> ref, float abs_tol,
                       std::string_view tag) {
  expect(got.size() == ref.size(), std::string(tag) + " size");
  float const d = max_abs_diff_bf16(got, ref);
  if (d > abs_tol) {
    fail(std::string(tag) + " max abs " + std::to_string(d) + " > " +
         std::to_string(abs_tol));
  }
}

void test_invalid_launches(Stream const& stream) {
  auto buf = DeviceBuffer::allocate(64);
  expect(static_cast<bool>(buf), "tiny buffer");
  if (!buf) {
    return;
  }
  auto bad_tok = qw38::cuda::launch_embed_gather(
      static_cast<std::uint16_t const*>(buf->data()), 4, 4,
      static_cast<float*>(buf->data()), stream);
  expect(!bad_tok && bad_tok.error().code == ErrorCode::InvalidArgument,
         "cuda embed invalid token");
  auto bad_eps = qw38::cuda::launch_hidden_rms(
      static_cast<float const*>(buf->data()),
      static_cast<std::uint16_t const*>(buf->data()), 0.0f, 1,
      static_cast<std::uint16_t*>(buf->data()), stream);
  expect(!bad_eps && bad_eps.error().code == ErrorCode::InvalidArgument,
         "cuda hidden rms eps");
  auto bad_pos = qw38::cuda::launch_partial_rope(
      static_cast<std::uint16_t const*>(buf->data()),
      static_cast<float const*>(buf->data()), -1, 1,
      static_cast<std::uint16_t*>(buf->data()), stream);
  expect(!bad_pos && bad_pos.error().code == ErrorCode::InvalidArgument,
         "cuda rope negative position");
  Stream empty;
  auto bad_stream = qw38::cuda::launch_silu_fp32(
      static_cast<float const*>(buf->data()), static_cast<float*>(buf->data()),
      4, empty);
  expect(!bad_stream && bad_stream.error().code == ErrorCode::InvalidArgument,
         "cuda empty stream");
}

void test_embed(Stream const& stream) {
  std::uint32_t const vocab = 3;
  auto table = ramp_h(vocab * kHidden, 1.5f);
  std::vector<float> cpu(kHidden);
  expect(static_cast<bool>(qw38::reference::embed_gather_bf16_to_fp32(
             table, vocab, 1, cpu)),
         "cpu embed");
  auto d_table = upload_vec(table, stream);
  auto d_out = DeviceBuffer::allocate(kHidden * sizeof(float));
  expect(static_cast<bool>(d_table) && static_cast<bool>(d_out), "embed alloc");
  if (!d_table || !d_out) {
    return;
  }
  auto st = qw38::cuda::launch_embed_gather(
      static_cast<std::uint16_t const*>(d_table->data()), vocab, 1,
      static_cast<float*>(d_out->data()), stream);
  expect(static_cast<bool>(st), "launch embed");
  auto got = download_vec<float>(*d_out, kHidden, stream);
  expect(static_cast<bool>(got), "download embed");
  if (!got) {
    return;
  }
  expect(*got == cpu, "embed CUDA matches CPU exactly");
}

void test_hidden_and_qk_rms(Stream const& stream) {
  auto x = ramp_f(kHidden, 4.0f, 0.1f);
  auto gamma = ramp_h(kHidden, 0.4f);
  std::vector<std::uint16_t> cpu(kHidden);
  std::vector<std::uint16_t> gold(kHidden);
  expect(static_cast<bool>(qw38::reference::hidden_rms_norm_1p_gamma(
             x, gamma, kDefaultRmsEps, cpu)),
         "cpu hidden rms");
  expect(static_cast<bool>(qw38::reference::hidden_rms_norm_1p_gamma_f64(
             x, gamma, kDefaultRmsEps, gold)),
         "f64 hidden rms");

  auto d_x = upload_vec(x, stream);
  auto d_g = upload_vec(gamma, stream);
  auto d_y = DeviceBuffer::allocate(kHidden * sizeof(std::uint16_t));
  expect(static_cast<bool>(d_x) && static_cast<bool>(d_g) && static_cast<bool>(d_y),
         "hidden rms alloc");
  if (!d_x || !d_g || !d_y) {
    return;
  }
  auto st = qw38::cuda::launch_hidden_rms(
      static_cast<float const*>(d_x->data()),
      static_cast<std::uint16_t const*>(d_g->data()), kDefaultRmsEps, 1,
      static_cast<std::uint16_t*>(d_y->data()), stream);
  expect(static_cast<bool>(st), "launch hidden rms");
  auto got = download_vec<std::uint16_t>(*d_y, kHidden, stream);
  expect(static_cast<bool>(got), "download hidden rms");
  if (!got) {
    return;
  }
  expect_bf16_close(*got, cpu, tol::kRmsBf16Abs, "hidden rms vs fp32");
  expect_bf16_close(*got, gold, tol::kRmsBf16Abs, "hidden rms vs f64");

  auto q = ramp_f(kHeadDim, 2.0f);
  auto qg = ramp_h(kHeadDim, 0.3f);
  std::vector<std::uint16_t> qcpu(kHeadDim);
  std::vector<std::uint16_t> qgold(kHeadDim);
  expect(static_cast<bool>(
             qw38::reference::qk_rms_norm_1p_gamma(q, qg, kDefaultRmsEps, qcpu)),
         "cpu qk");
  expect(static_cast<bool>(qw38::reference::qk_rms_norm_1p_gamma_f64(
             q, qg, kDefaultRmsEps, qgold)),
         "f64 qk");
  auto d_q = upload_vec(q, stream);
  auto d_qg = upload_vec(qg, stream);
  auto d_qo = DeviceBuffer::allocate(kHeadDim * sizeof(std::uint16_t));
  expect(static_cast<bool>(d_q) && static_cast<bool>(d_qg) &&
             static_cast<bool>(d_qo),
         "qk alloc");
  if (!d_q || !d_qg || !d_qo) {
    return;
  }
  auto qst = qw38::cuda::launch_qk_rms(
      static_cast<float const*>(d_q->data()),
      static_cast<std::uint16_t const*>(d_qg->data()), kDefaultRmsEps, 1,
      static_cast<std::uint16_t*>(d_qo->data()), stream);
  expect(static_cast<bool>(qst), "launch qk");
  auto qgot = download_vec<std::uint16_t>(*d_qo, kHeadDim, stream);
  expect(static_cast<bool>(qgot), "download qk");
  if (!qgot) {
    return;
  }
  expect_bf16_close(*qgot, qcpu, tol::kRmsBf16Abs, "qk vs fp32");
  expect_bf16_close(*qgot, qgold, tol::kRmsBf16Abs, "qk vs f64");
}

void test_gdn_gated(Stream const& stream) {
  auto o = ramp_f(kGdnHeadDim, 1.7f);
  auto z = ramp_h(kGdnHeadDim, 0.9f);
  auto g = ramp_h(kGdnHeadDim, 0.5f);
  std::vector<std::uint16_t> cpu(kGdnHeadDim);
  std::vector<std::uint16_t> gold(kGdnHeadDim);
  expect(static_cast<bool>(qw38::reference::gdn_gated_rms_norm(
             o, z, g, kDefaultRmsEps, cpu)),
         "cpu gated");
  expect(static_cast<bool>(qw38::reference::gdn_gated_rms_norm_f64(
             o, z, g, kDefaultRmsEps, gold)),
         "f64 gated");
  auto d_o = upload_vec(o, stream);
  auto d_z = upload_vec(z, stream);
  auto d_g = upload_vec(g, stream);
  auto d_y = DeviceBuffer::allocate(kGdnHeadDim * sizeof(std::uint16_t));
  expect(static_cast<bool>(d_o) && static_cast<bool>(d_z) && static_cast<bool>(d_g) &&
             static_cast<bool>(d_y),
         "gated alloc");
  if (!d_o || !d_z || !d_g || !d_y) {
    return;
  }
  auto st = qw38::cuda::launch_gdn_gated_rms(
      static_cast<float const*>(d_o->data()),
      static_cast<std::uint16_t const*>(d_z->data()),
      static_cast<std::uint16_t const*>(d_g->data()), kDefaultRmsEps, 1,
      static_cast<std::uint16_t*>(d_y->data()), stream);
  expect(static_cast<bool>(st), "launch gated");
  auto got = download_vec<std::uint16_t>(*d_y, kGdnHeadDim, stream);
  expect(static_cast<bool>(got), "download gated");
  if (!got) {
    return;
  }
  expect_bf16_close(*got, cpu, tol::kGatedRmsBf16Abs, "gated vs fp32");
  expect_bf16_close(*got, gold, tol::kGatedRmsBf16Abs, "gated vs f64");
}

void test_silu_sigmoid(Stream const& stream) {
  std::vector<float> in;
  for (int i = -40; i <= 40; ++i) {
    in.push_back(0.25f * static_cast<float>(i));
  }
  in.push_back(-80.0f);
  in.push_back(80.0f);
  std::vector<float> cpu_s(in.size());
  std::vector<float> cpu_l(in.size());
  std::vector<double> gold_s(in.size());
  std::vector<double> gold_l(in.size());
  expect(static_cast<bool>(qw38::reference::sigmoid_fp32(in, cpu_s)), "cpu sig");
  expect(static_cast<bool>(qw38::reference::silu_fp32(in, cpu_l)), "cpu silu");
  expect(static_cast<bool>(qw38::reference::sigmoid_f64(in, gold_s)), "f64 sig");
  expect(static_cast<bool>(qw38::reference::silu_f64(in, gold_l)), "f64 silu");
  auto d_in = upload_vec(in, stream);
  auto d_s = DeviceBuffer::allocate(in.size() * sizeof(float));
  auto d_l = DeviceBuffer::allocate(in.size() * sizeof(float));
  expect(static_cast<bool>(d_in) && static_cast<bool>(d_s) && static_cast<bool>(d_l),
         "silu alloc");
  if (!d_in || !d_s || !d_l) {
    return;
  }
  expect(static_cast<bool>(qw38::cuda::launch_sigmoid_fp32(
             static_cast<float const*>(d_in->data()),
             static_cast<float*>(d_s->data()),
             static_cast<std::uint32_t>(in.size()), stream)),
         "launch sigmoid");
  expect(static_cast<bool>(qw38::cuda::launch_silu_fp32(
             static_cast<float const*>(d_in->data()),
             static_cast<float*>(d_l->data()),
             static_cast<std::uint32_t>(in.size()), stream)),
         "launch silu");
  auto got_s = download_vec<float>(*d_s, in.size(), stream);
  auto got_l = download_vec<float>(*d_l, in.size(), stream);
  expect(static_cast<bool>(got_s) && static_cast<bool>(got_l), "download silu");
  if (!got_s || !got_l) {
    return;
  }
  for (std::size_t i = 0; i < in.size(); ++i) {
    expect(almost_equal((*got_s)[i], cpu_s[i], tol::kSigmoidAbs, 1.0e-5f),
           "sigmoid vs fp32");
    expect(almost_equal((*got_l)[i], cpu_l[i], tol::kSiluAbs, 1.0e-5f),
           "silu vs fp32");
    expect(almost_equal((*got_s)[i], static_cast<float>(gold_s[i]),
                        5.0e-6f, 1.0e-5f),
           "sigmoid vs f64");
    expect(almost_equal((*got_l)[i], static_cast<float>(gold_l[i]), 5.0e-6f,
                        1.0e-5f),
           "silu vs f64");
  }
}

void test_rope(Stream const& stream) {
  auto inv = qw38::reference::rope_inv_freq();
  auto head = ramp_h(kHeadDim, 1.1f);
  std::vector<std::int32_t> positions{0, 7, 1234, 262144};
  auto d_head = upload_vec(head, stream);
  auto d_inv = upload_vec(std::vector<float>(inv.begin(), inv.end()), stream);
  auto d_out = DeviceBuffer::allocate(kHeadDim * sizeof(std::uint16_t));
  expect(static_cast<bool>(d_head) && static_cast<bool>(d_inv) &&
             static_cast<bool>(d_out),
         "rope alloc");
  if (!d_head || !d_inv || !d_out) {
    return;
  }
  for (auto pos : positions) {
    std::vector<std::uint16_t> cpu(kHeadDim);
    std::vector<std::uint16_t> gold(kHeadDim);
    expect(static_cast<bool>(qw38::reference::partial_rope(head, inv, pos, cpu)),
           "cpu rope");
    expect(static_cast<bool>(
               qw38::reference::partial_rope_f64(head, inv, pos, gold)),
           "f64 rope");
    auto st = qw38::cuda::launch_partial_rope(
        static_cast<std::uint16_t const*>(d_head->data()),
        static_cast<float const*>(d_inv->data()), pos, 1,
        static_cast<std::uint16_t*>(d_out->data()), stream);
    expect(static_cast<bool>(st), "launch rope");
    auto got = download_vec<std::uint16_t>(*d_out, kHeadDim, stream);
    expect(static_cast<bool>(got), "download rope");
    if (!got) {
      return;
    }
    for (std::uint32_t i = qw38::reference::kRotaryDim; i < kHeadDim; ++i) {
      if ((*got)[i] != head[i]) {
        fail("cuda RoPE suffix unchanged");
        break;
      }
    }
    float abs_tol = (pos >= 100000) ? tol::kRopeLargeAbs : tol::kRopeSmallAbs;
    expect_bf16_close(*got, cpu, abs_tol, "rope vs fp32");
    // Double gold uses a different range-reduction; keep the large-pos budget.
    expect_bf16_close(*got, gold, abs_tol, "rope vs f64");
  }
}

void test_argmax(Stream const& stream) {
  std::vector<float> logits(1024);
  for (std::uint32_t i = 0; i < logits.size(); ++i) {
    logits[i] = 0.01f * static_cast<float>(static_cast<int>(i % 97) - 40);
  }
  logits[400] = 12.5f;
  logits[800] = 12.5f;  // later tie; first index 400 wins
  auto cpu = qw38::reference::argmax_fp32(logits);
  expect(static_cast<bool>(cpu) && *cpu == 400, "cpu argmax tie");
  auto d_in = upload_vec(logits, stream);
  auto d_out = DeviceBuffer::allocate(sizeof(std::uint32_t));
  expect(static_cast<bool>(d_in) && static_cast<bool>(d_out), "argmax alloc");
  if (!d_in || !d_out) {
    return;
  }
  auto st = qw38::cuda::launch_argmax_fp32(
      static_cast<float const*>(d_in->data()),
      static_cast<std::uint32_t>(logits.size()),
      static_cast<std::uint32_t*>(d_out->data()), stream);
  expect(static_cast<bool>(st), "launch argmax");
  auto got = download_vec<std::uint32_t>(*d_out, 1, stream);
  expect(static_cast<bool>(got) && (*got)[0] == 400, "cuda argmax matches");
}

void test_adversarial_magnitudes(Stream const& stream) {
  auto x = ramp_f(kHidden, 80.0f);
  x[0] = 1.0e-4f;
  x[1] = -70.0f;
  auto gamma = filled_h(kHidden, 0.0f);
  gamma[2] = bf16(3.0f);
  std::vector<std::uint16_t> cpu(kHidden);
  expect(static_cast<bool>(qw38::reference::hidden_rms_norm_1p_gamma(
             x, gamma, kDefaultRmsEps, cpu)),
         "cpu adversarial rms");
  auto d_x = upload_vec(x, stream);
  auto d_g = upload_vec(gamma, stream);
  auto d_y = DeviceBuffer::allocate(kHidden * sizeof(std::uint16_t));
  if (!d_x || !d_g || !d_y) {
    fail("adv alloc");
    return;
  }
  expect(static_cast<bool>(qw38::cuda::launch_hidden_rms(
             static_cast<float const*>(d_x->data()),
             static_cast<std::uint16_t const*>(d_g->data()), kDefaultRmsEps, 1,
             static_cast<std::uint16_t*>(d_y->data()), stream)),
         "launch adv rms");
  auto got = download_vec<std::uint16_t>(*d_y, kHidden, stream);
  expect(static_cast<bool>(got), "download adv rms");
  if (got) {
    expect_bf16_close(*got, cpu, tol::kRmsBf16Abs, "adversarial rms");
  }
}

}  // namespace

int main() {
  auto stream = Stream::create();
  expect(static_cast<bool>(stream), "create stream");
  if (!stream) {
    std::cerr << g_failures << " failures\n";
    return 1;
  }
  test_invalid_launches(*stream);
  test_embed(*stream);
  test_hidden_and_qk_rms(*stream);
  test_gdn_gated(*stream);
  test_silu_sigmoid(*stream);
  test_rope(*stream);
  test_argmax(*stream);
  test_adversarial_magnitudes(*stream);
  if (g_failures != 0) {
    std::cerr << g_failures << " failures\n";
    return 1;
  }
  return 0;
}
