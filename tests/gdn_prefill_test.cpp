#include "gdn_support.hpp"

#include "cuda/copy.hpp"
#include "cuda/gdn.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

namespace {

using namespace qw38::gdn::test;
using qw38::cuda::DeviceBuffer;
using qw38::cuda::launch_gdn_prefill_conv;
using qw38::cuda::launch_gdn_prefill_history_commit;
using qw38::cuda::launch_gdn_prefill_prepare;
using qw38::cuda::launch_gdn_prefill_recurrence;
using qw38::cuda::launch_gdn_conv_silu;
using qw38::cuda::launch_gdn_recurrence;
using qw38::format::bf16_to_fp32;

template <class T>
std::vector<T> download(void const* ptr, std::size_t count, Stream const& stream) {
  std::vector<T> out(count);
  auto st = qw38::cuda::copy_d2h(out.data(), ptr, count * sizeof(T), stream);
  if (!st || !stream.sync()) { fail("device download or sync"); return {}; }
  return out;
}

void fir_cases(Stream const& stream) {
  auto taps = pattern_h(kConvKernel * kQkvWidth, 0.12f);
  auto dt = upload_vec(taps, stream);
  if (!dt) { fail("FIR taps upload"); return; }
  for (std::uint32_t cursor : {0u, 1u, 2u}) {
    for (std::uint32_t count : {1u, 2u, 3u, 4u, 7u}) {
      std::vector<std::uint16_t> qkv(count * kQkvWidth);
      std::vector<std::uint16_t> history(3u * kQkvWidth);
      for (std::size_t i = 0; i < qkv.size(); ++i)
        qkv[i] = bf16(0.03f * static_cast<float>(static_cast<int>(i % 29u) - 14));
      for (std::size_t i = 0; i < history.size(); ++i)
        history[i] = bf16(0.025f * static_cast<float>(static_cast<int>(i % 23u) - 11));
      auto expected_history = history;
      auto expected_cursor = cursor;
      std::vector<std::uint16_t> expected(count * kQkvWidth);
      for (std::uint32_t t = 0; t < count; ++t) {
        auto st = qw38::reference::gdn_conv_history_step(
            std::span<std::uint16_t const>{qkv.data() + t * kQkvWidth, kQkvWidth},
            taps, expected_history, expected_cursor,
            std::span<std::uint16_t>{expected.data() + t * kQkvWidth, kQkvWidth});
        expect(bool(st), "independent FIR step");
      }
      auto dq = upload_vec(qkv, stream);
      auto dh = upload_vec(history, stream);
      auto dc = DeviceBuffer::allocate((count + 1u) * kQkvWidth * 2u);
      if (!dq || !dh || !dc) { fail("FIR allocations"); return; }
      if (!qw38::cuda::zero(*dc, stream)) { fail("FIR output zero"); return; }
      auto st = launch_gdn_prefill_conv(
          static_cast<std::uint16_t const*>(dq->data()),
          static_cast<std::uint16_t const*>(dt->data()),
          static_cast<std::uint16_t const*>(dh->data()), cursor,
          static_cast<std::uint16_t*>(dc->data()), count, stream);
      expect(bool(st), "prefill FIR launch");
      st = launch_gdn_prefill_history_commit(
          static_cast<std::uint16_t const*>(dq->data()),
          static_cast<std::uint16_t*>(dh->data()), cursor, count, stream);
      expect(bool(st), "prefill history commit launch");
      auto got = download<std::uint16_t>(dc->data(), count * kQkvWidth, stream);
      auto guard = download<std::uint16_t>(
          static_cast<std::uint16_t const*>(dc->data()) + count * kQkvWidth,
          kQkvWidth, stream);
      expect(guard.size() == kQkvWidth && std::all_of(guard.begin(), guard.end(),
                         [](std::uint16_t x) { return x == 0u; }),
             "FIR valid-row mask");
      auto got_history = download<std::uint16_t>(dh->data(), 3u * kQkvWidth, stream);
      expect(got_history == expected_history, "history order and short-tail preservation");
      expect(got.size() == expected.size(), "parallel FIR download size");
      if (got.size() == expected.size())
        expect_bf16_close(got, expected, "parallel FIR", kGdnConvBf16Abs, 0.0f);
      auto next_qkv = pattern_h(kQkvWidth, 0.31f);
      auto next_expected = zeros_h(kQkvWidth);
      auto next_host = qw38::reference::gdn_conv_history_step(
          next_qkv, taps, expected_history, expected_cursor, next_expected);
      auto dn = upload_vec(next_qkv, stream);
      if (!next_host || !dn) { fail("FIR continuation setup"); return; }
      st = launch_gdn_conv_silu(static_cast<std::uint16_t const*>(dn->data()),
          static_cast<std::uint16_t const*>(dt->data()),
          static_cast<std::uint16_t*>(dh->data()), (cursor + count) % 3u,
          static_cast<std::uint16_t*>(dc->data()), stream);
      expect(bool(st), "prefill-to-decode FIR launch");
      auto next_got = download<std::uint16_t>(dc->data(), kQkvWidth, stream);
      expect(next_got.size() == next_expected.size(), "FIR continuation download size");
      if (next_got.size() == next_expected.size())
        expect_bf16_close(next_got, next_expected, "FIR continuation",
                          kGdnConvBf16Abs, 0.0f);
    }
  }
}

void wy_row_reference(std::vector<float> const& q, std::vector<float> const& k,
                      std::vector<float> const& alpha,
                      std::vector<float> const& beta,
                      std::vector<std::uint16_t> const& convolved,
                      std::vector<float> const& initial,
                      std::vector<float> const& serial_state,
                      std::vector<float> const& serial_o,
                      std::uint32_t count) {
  // Independent affine composition of the same map, for one value row:
  // S_t = A_t S_(t-1) + b_t, A_t = alpha_t(I-beta_t k_t k_t^T),
  // b_t = beta_t v_t k_t. P and c are the chunk transform S_t=P S_0+c.
  constexpr std::uint32_t d = 128, vh = 7, value = 11;
  std::vector<double> p(d * d, 0.0), next(d * d), c(d, 0.0), next_c(d);
  for (std::uint32_t i = 0; i < d; ++i) p[i * d + i] = 1.0;
  std::size_t const row = (static_cast<std::size_t>(vh) * d + value) * d;
  for (std::uint32_t t = 0; t < count; ++t) {
    std::size_t const qbase = (static_cast<std::size_t>(t) * 16u + vh / 3u) * d;
    double const a = alpha[static_cast<std::size_t>(t) * 48u + vh];
    double const b = beta[static_cast<std::size_t>(t) * 48u + vh];
    double const v = bf16_to_fp32(convolved[
        static_cast<std::size_t>(t) * kQkvWidth + 4096u + vh * d + value]);
    for (std::uint32_t i = 0; i < d; ++i) {
      double dot = 0.0;
      for (std::uint32_t h = 0; h < d; ++h) dot += k[qbase + h] * p[h * d + i];
      for (std::uint32_t h = 0; h < d; ++h)
        next[h * d + i] = a * (p[h * d + i] - b * k[qbase + h] * dot);
    }
    double kc = 0.0;
    for (std::uint32_t h = 0; h < d; ++h) kc += k[qbase + h] * c[h];
    for (std::uint32_t h = 0; h < d; ++h)
      next_c[h] = a * (c[h] - b * k[qbase + h] * kc) + b * v * k[qbase + h];
    p.swap(next); c.swap(next_c);
  }
  std::vector<double> transformed(d);
  for (std::uint32_t h = 0; h < d; ++h) {
    double result = c[h];
    for (std::uint32_t i = 0; i < d; ++i)
      result += p[h * d + i] * initial[row + i];
    transformed[h] = result;
    expect(std::fabs(result - serial_state[row + h]) < 8.0e-4,
           "independent WY affine final state");
  }
  double output = 0.0;
  std::size_t const qbase = (static_cast<std::size_t>(count - 1u) * 16u + vh / 3u) * d;
  for (std::uint32_t h = 0; h < d; ++h)
    output += transformed[h] * q[qbase + h] / std::sqrt(128.0);
  expect(std::fabs(output - serial_o[
      (static_cast<std::size_t>(count - 1u) * 48u + vh) * d + value]) < 8.0e-4,
      "independent WY output contraction");
}

void recurrence_case(Stream const& stream) {
  constexpr std::uint32_t count = 65;
  std::vector<std::uint16_t> convolved(count * kQkvWidth);
  for (std::size_t i = 0; i < convolved.size(); ++i)
    convolved[i] = bf16(0.09f * static_cast<float>(static_cast<int>(i % 17u) - 8));
  std::vector<float> a(count * 48u), b(count * 48u);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<float>(static_cast<int>(i % 11u) - 5) * 0.42f;
    b[i] = static_cast<float>(static_cast<int>(i % 13u) - 6) * 0.68f;
  }
  auto a_log = pattern_h(48u, -0.3f);
  auto dt = pattern_h(48u, 0.2f);
  std::vector<float> q(count * 2048u), k(count * 2048u);
  std::vector<float> alpha(count * 48u), beta(count * 48u);
  for (std::uint32_t t = 0; t < count; ++t) {
    auto st = qw38::reference::gdn_prepare(
        std::span<std::uint16_t const>{convolved.data() + t * kQkvWidth, kQkvWidth},
        std::span<float const>{a.data() + t * 48u, 48u},
        std::span<float const>{b.data() + t * 48u, 48u},
        a_log, dt, 1.0e-6f,
        std::span<float>{q.data() + t * 2048u, 2048u},
        std::span<float>{k.data() + t * 2048u, 2048u},
        std::span<float>{alpha.data() + t * 48u, 48u},
        std::span<float>{beta.data() + t * 48u, 48u});
    expect(bool(st), "independent prepare");
  }
  auto initial = pattern_f(static_cast<std::uint32_t>(kGdnSElemsPerLayer), 0.03f);
  auto serial = initial;
  std::vector<float> serial_7;
  std::vector<float> serial_o(count * 6144u);
  for (std::uint32_t t = 0; t < count; ++t) {
    auto st = qw38::reference::gdn_recurrence_step(
        std::span<float const>{q.data() + t * 2048u, 2048u},
        std::span<float const>{k.data() + t * 2048u, 2048u},
        std::span<float const>{alpha.data() + t * 48u, 48u},
        std::span<float const>{beta.data() + t * 48u, 48u},
        std::span<std::uint16_t const>{convolved.data() + t * kQkvWidth + 4096u, 6144u},
        serial, std::span<float>{serial_o.data() + t * 6144u, 6144u});
    expect(bool(st), "independent serial recurrence");
    if (t == 6u) serial_7 = serial;
  }
  wy_row_reference(q, k, alpha, beta, convolved, initial, serial_7, serial_o, 7u);
  auto dc = upload_vec(convolved, stream), da = upload_vec(a, stream);
  auto db = upload_vec(b, stream), dal = upload_vec(a_log, stream);
  auto ddt = upload_vec(dt, stream), ds = upload_vec(initial, stream);
  auto dq = DeviceBuffer::allocate(q.size() * 4u), dk = DeviceBuffer::allocate(k.size() * 4u);
  auto d_alpha = DeviceBuffer::allocate(alpha.size() * 4u);
  auto d_beta = DeviceBuffer::allocate(beta.size() * 4u);
  auto dout = DeviceBuffer::allocate((serial_o.size() + 6144u) * 4u);
  if (!dc || !da || !db || !dal || !ddt || !ds || !dq || !dk ||
      !d_alpha || !d_beta || !dout) { fail("recurrence allocations"); return; }
  auto st = launch_gdn_prefill_prepare(
      static_cast<std::uint16_t const*>(dc->data()),
      static_cast<float const*>(da->data()), static_cast<float const*>(db->data()),
      static_cast<std::uint16_t const*>(dal->data()),
      static_cast<std::uint16_t const*>(ddt->data()), 1.0e-6f,
      static_cast<float*>(dq->data()), static_cast<float*>(dk->data()),
      static_cast<float*>(d_alpha->data()), static_cast<float*>(d_beta->data()),
      count, stream);
  expect(bool(st), "parallel prepare launch");
  auto got_q = download<float>(dq->data(), q.size(), stream);
  auto got_k = download<float>(dk->data(), k.size(), stream);
  auto got_a = download<float>(d_alpha->data(), alpha.size(), stream);
  auto got_b = download<float>(d_beta->data(), beta.size(), stream);
  expect(got_q.size() == q.size() && got_k.size() == k.size() &&
         got_a.size() == alpha.size() && got_b.size() == beta.size(),
         "prepare download sizes");
  if (got_q.size() == q.size()) expect_fp32_close(got_q, q, "prefill q", 2e-5f, 1e-5f);
  if (got_k.size() == k.size()) expect_fp32_close(got_k, k, "prefill k", 2e-5f, 1e-5f);
  if (got_a.size() == alpha.size()) expect_fp32_close(got_a, alpha, "prefill alpha", 2e-5f, 1e-5f);
  if (got_b.size() == beta.size()) expect_fp32_close(got_b, beta, "prefill beta", 2e-5f, 1e-5f);
  for (std::uint32_t interval : {1u, 3u, 7u, 64u, 65u}) {
    if (!qw38::cuda::zero(*dout, stream)) { fail("recurrence output zero"); return; }
    auto reset = qw38::cuda::copy_h2d(ds->data(), initial.data(),
                                      initial.size() * 4u, stream);
    expect(bool(reset), "state reset copy");
    st = launch_gdn_prefill_recurrence(
        static_cast<float const*>(dq->data()), static_cast<float const*>(dk->data()),
        static_cast<float const*>(d_alpha->data()),
        static_cast<float const*>(d_beta->data()),
        static_cast<std::uint16_t const*>(dc->data()),
        static_cast<float*>(ds->data()), 0, static_cast<float*>(dout->data()),
        count, interval, stream);
    expect(bool(st), "state-resident recurrence launch");
    auto got_state = download<float>(ds->data(), initial.size(), stream);
    auto got_o = download<float>(dout->data(), serial_o.size(), stream);
    auto guard = download<float>(static_cast<float const*>(dout->data()) +
                                     serial_o.size(), 6144u, stream);
    expect(guard.size() == 6144u && std::all_of(guard.begin(), guard.end(),
                       [](float x) { return x == 0.0f; }),
           "recurrence valid-row mask");
    expect(got_state.size() == serial.size() && got_o.size() == serial_o.size(),
           "recurrence download sizes");
    if (got_state.size() == serial.size())
      expect_fp32_close(got_state, serial, "prefill recurrence state", 7e-4f, 2e-4f);
    if (got_o.size() == serial_o.size())
      expect_fp32_close(got_o, serial_o, "prefill recurrence output", 7e-4f, 2e-4f);
  }
  std::vector<float> continuation_o(6144u);
  auto next_serial = serial;
  auto ref_step = qw38::reference::gdn_recurrence_step(
      std::span<float const>{q.data() + (count - 1u) * 2048u, 2048u},
      std::span<float const>{k.data() + (count - 1u) * 2048u, 2048u},
      std::span<float const>{alpha.data() + (count - 1u) * 48u, 48u},
      std::span<float const>{beta.data() + (count - 1u) * 48u, 48u},
      std::span<std::uint16_t const>{convolved.data() +
          (count - 1u) * kQkvWidth + 4096u, 6144u},
      next_serial, continuation_o);
  expect(bool(ref_step), "decode continuation reference");
  st = launch_gdn_recurrence(
      static_cast<float const*>(dq->data()) + (count - 1u) * 2048u,
      static_cast<float const*>(dk->data()) + (count - 1u) * 2048u,
      static_cast<float const*>(d_alpha->data()) + (count - 1u) * 48u,
      static_cast<float const*>(d_beta->data()) + (count - 1u) * 48u,
      static_cast<std::uint16_t const*>(dc->data()) +
          (count - 1u) * kQkvWidth + 4096u,
      static_cast<float*>(ds->data()), 0, static_cast<float*>(dout->data()), stream);
  expect(bool(st), "prefill-to-decode recurrence launch");
  auto decoded_state = download<float>(ds->data(), initial.size(), stream);
  auto decoded_o = download<float>(dout->data(), continuation_o.size(), stream);
  expect(decoded_state.size() == next_serial.size() &&
         decoded_o.size() == continuation_o.size(), "decode continuation download sizes");
  if (decoded_state.size() == next_serial.size())
    expect_fp32_close(decoded_state, next_serial, "state continuation", 8e-4f, 2e-4f);
  if (decoded_o.size() == continuation_o.size())
    expect_fp32_close(decoded_o, continuation_o, "output continuation", 8e-4f, 2e-4f);
}

void boundary_and_long_case(Stream const& stream) {
  constexpr std::uint32_t count = 193;
  std::vector<float> q(count * 2048u), k(count * 2048u);
  std::vector<float> alpha(count * 48u), beta(count * 48u);
  std::vector<std::uint16_t> convolved(count * kQkvWidth);
  auto initial = pattern_f(static_cast<std::uint32_t>(kGdnSElemsPerLayer), 0.11f);
  for (std::uint32_t t = 0; t < count; ++t) {
    for (std::uint32_t h = 0; h < 16u; ++h)
      for (std::uint32_t j = 0; j < 128u; ++j) {
        auto const i = (static_cast<std::size_t>(t) * 16u + h) * 128u + j;
        float const sign = ((t + h + j) & 1u) ? -1.0f : 1.0f;
        q[i] = sign * (j == (t % 128u) ? 0.7f : 0.04f);
        k[i] = sign * (j == ((t + 1u) % 128u) ? 0.7f : 0.04f);
      }
    for (std::uint32_t h = 0; h < 48u; ++h) {
      auto const i = static_cast<std::size_t>(t) * 48u + h;
      alpha[i] = (t % 3u == 0u) ? 0.999999f :
                 (t % 3u == 1u) ? 0.0001f : 0.75f;
      beta[i] = (t % 4u == 0u) ? 0.0f :
                (t % 4u == 1u) ? 0.9999f : 0.5f;
    }
    for (std::uint32_t j = 0; j < kQkvWidth; ++j)
      convolved[static_cast<std::size_t>(t) * kQkvWidth + j] =
          bf16(((t + j) & 1u ? -1.0f : 1.0f) *
               (j % 7u == 0u ? 0.6f : 0.03f));
  }
  auto serial = initial;
  std::vector<float> expected(count * 6144u);
  for (std::uint32_t t = 0; t < count; ++t)
    expect(bool(qw38::reference::gdn_recurrence_step(
        std::span<float const>{q.data() + t * 2048u, 2048u},
        std::span<float const>{k.data() + t * 2048u, 2048u},
        std::span<float const>{alpha.data() + t * 48u, 48u},
        std::span<float const>{beta.data() + t * 48u, 48u},
        std::span<std::uint16_t const>{convolved.data() +
            static_cast<std::size_t>(t) * kQkvWidth + 4096u, 6144u},
        serial, std::span<float>{expected.data() + t * 6144u, 6144u})),
        "long serial reference");
  auto dq = upload_vec(q, stream), dk = upload_vec(k, stream);
  auto da = upload_vec(alpha, stream), db = upload_vec(beta, stream);
  auto dc = upload_vec(convolved, stream), ds = upload_vec(initial, stream);
  auto dout = DeviceBuffer::allocate(expected.size() * 4u);
  if (!dq || !dk || !da || !db || !dc || !ds || !dout) {
    fail("long recurrence allocations"); return;
  }
  auto run = [&](std::uint32_t first, std::uint32_t n) {
    return launch_gdn_prefill_recurrence(
        static_cast<float const*>(dq->data()) + first * 2048u,
        static_cast<float const*>(dk->data()) + first * 2048u,
        static_cast<float const*>(da->data()) + first * 48u,
        static_cast<float const*>(db->data()) + first * 48u,
        static_cast<std::uint16_t const*>(dc->data()) +
            static_cast<std::size_t>(first) * kQkvWidth,
        static_cast<float*>(ds->data()), 0,
        static_cast<float*>(dout->data()) + first * 6144u, n, 64u, stream);
  };
  expect(bool(run(0, 64)) && bool(run(64, 64)) && bool(run(128, 65)),
         "long chunked recurrence launch");
  auto got_state = download<float>(ds->data(), serial.size(), stream);
  auto got_out = download<float>(dout->data(), expected.size(), stream);
  expect(got_state.size() == serial.size() && got_out.size() == expected.size(),
         "long recurrence download sizes");
  if (got_state.size() == serial.size())
    expect_fp32_close(got_state, serial, "long recurrence state", 1e-3f, 3e-4f);
  if (got_out.size() == expected.size())
    expect_fp32_close(got_out, expected, "long recurrence output", 1e-3f, 3e-4f);

  auto const* qptr = static_cast<float const*>(dq->data());
  auto const* kptr = static_cast<float const*>(dk->data());
  auto const* aptr = static_cast<float const*>(da->data());
  auto const* bptr = static_cast<float const*>(db->data());
  auto const* cptr = static_cast<std::uint16_t const*>(dc->data());
  auto* sptr = static_cast<float*>(ds->data());
  auto* optr = static_cast<float*>(dout->data());
  expect(!launch_gdn_prefill_recurrence(qptr, kptr, aptr, bptr, cptr,
      sptr, 0, sptr, 1, 1, stream), "state/output alias rejection");
  expect(!launch_gdn_prefill_recurrence(qptr, kptr, aptr, bptr, cptr,
      sptr, 0, sptr + 1, 1, 1, stream), "partial state/output overlap rejection");
  expect(!launch_gdn_prefill_recurrence(qptr, kptr, aptr, bptr, cptr,
      sptr, 0, optr + expected.size() - 1u, 2, 1, stream),
      "undersized output storage rejection");
  expect(!launch_gdn_prefill_recurrence(q.data(), kptr, aptr, bptr, cptr,
      sptr, 0, optr, 1, 1, stream), "host pointer rejection");
  expect(!launch_gdn_prefill_conv(cptr, cptr, cptr, 0,
      const_cast<std::uint16_t*>(cptr), 1, stream), "FIR alias rejection");
  expect(!launch_gdn_prefill_prepare(cptr, aptr, bptr, cptr, cptr, 1e-6f,
      optr, optr + 1, optr + 2, optr + 3, 1, stream),
      "prepare partial overlap rejection");
}

}  // namespace

int main() {
  auto stream = Stream::create();
  if (!stream) { std::cerr << qw38::cuda::error_message(stream.error()) << '\n'; return 1; }
  fir_cases(*stream);
  recurrence_case(*stream);
  boundary_and_long_case(*stream);
  if (g_failures) std::cerr << "GDN prefill failures=" << g_failures << '\n';
  return g_failures ? 1 : 0;
}
