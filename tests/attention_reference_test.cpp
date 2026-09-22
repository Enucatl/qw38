#include "attention_support.hpp"

#include "cuda/copy.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

using qw38::attn::test::DeviceAttn;
using qw38::attn::test::HostAttn;
using qw38::attn::test::bind_views;
using qw38::attn::test::download_vec;
using qw38::attn::test::expect;
using qw38::attn::test::expect_bf16_close;
using qw38::attn::test::fail;
using qw38::attn::test::g_failures;
using qw38::attn::test::kAttnKvWidth;
using qw38::attn::test::kAttnPrepLargeAbs;
using qw38::attn::test::kAttnPrepLargeRel;
using qw38::attn::test::kAttnPrepSmallAbs;
using qw38::attn::test::kAttnPrepSmallRel;
using qw38::attn::test::kAttnProjAbs;
using qw38::attn::test::kAttnProjRel;
using qw38::attn::test::kHeadDim;
using qw38::attn::test::kHidden;
using qw38::attn::test::kKvHeads;
using qw38::attn::test::kQgWidth;
using qw38::attn::test::kQueryHeads;
using qw38::attn::test::kRotaryDim;
using qw38::attn::test::kv_index;
using qw38::attn::test::make_host_bf16;
using qw38::attn::test::make_host_q4;
using qw38::attn::test::upload_host_attn;
using qw38::cuda::Stream;
using qw38::reference::attn_prep_reference;
using qw38::reference::attn_online_core;
using qw38::runtime::bind_attention_prep_plan;
using qw38::runtime::bind_attention_workspace;
using qw38::runtime::execute_decode_attention_prep;
using qw38::runtime::kAttnOffG;
using qw38::runtime::kAttnOffK;
using qw38::runtime::kAttnOffQ;
using qw38::runtime::kAttnOffQg;
using qw38::runtime::kAttnOffV;

namespace {

bool compare_online_lengths() {
  constexpr std::uint64_t capacity = 257;
  constexpr std::uint64_t kv_elems = 16ull * 2ull * kKvHeads * capacity * kHeadDim;
  std::vector<std::uint16_t> q(kQueryHeads * kHeadDim, 0);
  std::vector<std::uint16_t> g(kQueryHeads * kHeadDim, 0);
  std::vector<std::uint16_t> kv(kv_elems, 0);
  auto kv_index = [&](std::uint32_t component, std::uint32_t h,
                      std::uint64_t t, std::uint32_t d) {
    auto const hs = capacity * kHeadDim;
    auto const cs = kKvHeads * hs;
    return component * cs + h * hs + t * kHeadDim + d;
  };
  for (std::uint32_t h = 0; h < kKvHeads; ++h) {
    for (std::uint64_t t = 0; t < capacity; ++t) {
      auto const bits = qw38::format::fp32_to_bf16_rne(
          0.01f * static_cast<float>(h + 1u) * static_cast<float>(t + 1u));
      for (std::uint32_t d = 0; d < kHeadDim; ++d) {
        kv[kv_index(1, h, t, d)] = bits;
      }
    }
  }
  for (std::uint64_t length : {0ull, 1ull, 31ull, 32ull, 255ull, 256ull,
                               257ull}) {
    auto segmented = attn_online_core(q, g, kv, 0, capacity, length, true);
    auto unsegmented = attn_online_core(q, g, kv, 0, capacity, length, false);
    if (!segmented || !unsegmented) {
      fail("online reference bind length " + std::to_string(length));
      return false;
    }
    for (std::size_t i = 0; i < segmented->attn.size(); ++i) {
      float const a = segmented->attn[i];
      float const b = unsegmented->attn[i];
      float const scale = std::fmax(1.0f, std::fabs(b));
      if (std::fabs(a - b) > 2.0e-5f * scale ||
          segmented->y[i] != unsegmented->y[i]) {
        fail("segmented/unsegmented mismatch length " + std::to_string(length));
        return false;
      }
    }
  }
  return true;
}

bool compare_prep(DeviceAttn& dev, HostAttn const& host, Stream const& stream,
                  std::int32_t position, std::string_view tag) {
  auto views = bind_views(dev, 3);
  auto plan = bind_attention_prep_plan(views, stream);
  if (!plan) {
    fail(std::string(tag) + " bind: " +
         qw38::runtime::error_message(plan.error()));
    return false;
  }
  dev.populated = static_cast<std::uint64_t>(position);
  auto st = execute_decode_attention_prep(*plan, static_cast<std::uint64_t>(position));
  if (!st) {
    fail(std::string(tag) + " execute: " + qw38::runtime::error_message(st.error()));
    return false;
  }
  auto cpu = attn_prep_reference(host.residual, host.gamma, qw38::reference::kDefaultRmsEps,
                                 host.w_qg, host.w_k, host.w_v, host.gamma_q,
                                 host.gamma_k, host.inv_freq, position);
  if (!cpu) {
    fail(std::string(tag) + " cpu: " + qw38::reference::error_message(cpu.error()));
    return false;
  }
  auto scratch = bind_attention_workspace(views.workspace);
  if (!scratch) {
    fail(std::string(tag) + " workspace");
    return false;
  }
  std::vector<std::uint16_t> qg_h(kQgWidth);
  std::vector<std::uint16_t> k_h(kAttnKvWidth);
  std::vector<std::uint16_t> v_h(kAttnKvWidth);
  std::vector<std::uint16_t> q_h(kQueryHeads * kHeadDim);
  std::vector<std::uint16_t> g_h(kQueryHeads * kHeadDim);
  auto copy_off = [&](std::vector<std::uint16_t>& dst, std::uint64_t off) {
    return qw38::cuda::copy_d2h(dst.data(),
                                static_cast<std::byte*>(dev.workspace.data()) + off,
                                dst.size() * 2, stream);
  };
  expect(static_cast<bool>(copy_off(qg_h, kAttnOffQg)), std::string(tag) + " qg d2h");
  expect(static_cast<bool>(copy_off(k_h, kAttnOffK)), std::string(tag) + " k d2h");
  expect(static_cast<bool>(copy_off(v_h, kAttnOffV)), std::string(tag) + " v d2h");
  expect(static_cast<bool>(copy_off(q_h, kAttnOffQ)), std::string(tag) + " Q d2h");
  expect(static_cast<bool>(copy_off(g_h, kAttnOffG)), std::string(tag) + " g d2h");
  if (auto s = stream.sync(); !s) {
    fail(std::string(tag) + " sync");
    return false;
  }
  float const prep_abs =
      position >= 10000 ? kAttnPrepLargeAbs : kAttnPrepSmallAbs;
  float const prep_rel =
      position >= 10000 ? kAttnPrepLargeRel : kAttnPrepSmallRel;
  expect_bf16_close(qg_h, cpu->qg, std::string(tag) + " projected qg", kAttnProjAbs,
                    kAttnProjRel);
  expect_bf16_close(k_h, cpu->k_raw, std::string(tag) + " projected k", kAttnProjAbs,
                    kAttnProjRel);
  expect_bf16_close(v_h, cpu->v_raw, std::string(tag) + " projected v", kAttnProjAbs,
                    kAttnProjRel);
  expect_bf16_close(q_h, cpu->q, std::string(tag) + " prepared Q", prep_abs, prep_rel);
  expect_bf16_close(g_h, cpu->g, std::string(tag) + " prepared g", kAttnProjAbs,
                    kAttnProjRel);

  auto kv = download_vec<std::uint16_t>(
      dev.kv, static_cast<std::size_t>(2u * kKvHeads * dev.capacity * kHeadDim),
      stream);
  if (!kv) {
    fail(std::string(tag) + " kv download");
    return false;
  }
  for (std::uint32_t h = 0; h < kKvHeads; ++h) {
    std::vector<std::uint16_t> k_got(kHeadDim);
    std::vector<std::uint16_t> v_got(kHeadDim);
    for (std::uint32_t i = 0; i < kHeadDim; ++i) {
      k_got[i] = (*kv)[kv_index(0, 0, h, static_cast<std::uint64_t>(position), i,
                                dev.capacity)];
      v_got[i] = (*kv)[kv_index(0, 1, h, static_cast<std::uint64_t>(position), i,
                                dev.capacity)];
    }
    expect_bf16_close(k_got,
                      std::span<std::uint16_t const>(cpu->k.data() + h * kHeadDim,
                                                     kHeadDim),
                      std::string(tag) + " cached K", prep_abs, prep_rel);
    expect_bf16_close(v_got,
                      std::span<std::uint16_t const>(cpu->v.data() + h * kHeadDim,
                                                     kHeadDim),
                      std::string(tag) + " cached V", kAttnProjAbs, kAttnProjRel);
    for (std::uint32_t i = kRotaryDim; i < kHeadDim; ++i) {
      float const got = qw38::format::bf16_to_fp32(k_got[i]);
      // Suffix of prepared K must match RMS of projected k, already covered by
      // the fused reference comparison above; keep a direct Q suffix check.
      (void)got;
    }
  }
  for (std::uint32_t h = 0; h < kQueryHeads; ++h) {
    for (std::uint32_t i = kRotaryDim; i < kHeadDim; ++i) {
      float const got = qw38::format::bf16_to_fp32(q_h[h * kHeadDim + i]);
      float const ref = qw38::format::bf16_to_fp32(cpu->q[h * kHeadDim + i]);
      float const d = got > ref ? got - ref : ref - got;
      if (d > prep_abs) {
        fail(std::string(tag) + " Q RoPE suffix");
        return false;
      }
    }
  }
  return true;
}

void run_family(bool q4, std::string_view name, Stream const& stream) {
  std::int32_t const positions[] = {0, 1, 123456};
  for (auto pos : positions) {
    HostAttn host;
    bool made = q4 ? make_host_q4(host, 5) : make_host_bf16(host, 0.2f);
    if (!made) {
      return;
    }
    DeviceAttn dev;
    if (!upload_host_attn(host, dev, stream, 131072)) {
      return;
    }
    std::string tag = std::string(name) + " pos " + std::to_string(pos);
    compare_prep(dev, host, stream, pos, tag);
  }
}

}  // namespace

int main() {
  auto stream = Stream::create();
  if (!stream) {
    fail("stream");
    return 1;
  }
  run_family(true, "q4", *stream);
  run_family(false, "bf16-control", *stream);
  expect(compare_online_lengths(), "segmented online reference boundaries");
  if (g_failures != 0) {
    std::cerr << g_failures << " attention reference failures\n";
    return 1;
  }
  std::cout << "attention reference ok\n";
  return 0;
}
