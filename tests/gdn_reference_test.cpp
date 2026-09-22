#include "gdn_support.hpp"

#include "compiler/quantization/reference.hpp"
#include "cuda/copy.hpp"
#include "cuda/decode_mmv.hpp"
#include "cuda/buffer.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using qw38::cuda::DeviceBuffer;
using qw38::cuda::Stream;
using qw38::cuda::launch_gdn_conv_silu;
using qw38::cuda::launch_gdn_prepare;
using qw38::cuda::launch_gdn_recurrence;
using qw38::format::LogicalQuantizerId;
using qw38::format::PackedMatrix;
using qw38::format::PhysicalLayoutId;
using qw38::format::StorageClass;
using qw38::gdn::test::DeviceGdnMixer;
using qw38::gdn::test::HostGdnMixer;
using qw38::gdn::test::download_vec;
using qw38::gdn::test::expect;
using qw38::gdn::test::expect_bf16_close;
using qw38::gdn::test::expect_fp32_close;
using qw38::gdn::test::fail;
using qw38::gdn::test::fill_logical_pattern;
using qw38::gdn::test::g_failures;
using qw38::gdn::test::kConvHistoryTaps;
using qw38::gdn::test::kConvKernel;
using qw38::gdn::test::kDefaultRmsEps;
using qw38::gdn::test::kGdnConvBf16Abs;
using qw38::gdn::test::kGdnGateFp32Abs;
using qw38::gdn::test::kGdnGateFp32Rel;
using qw38::gdn::test::kGdnHeadDim;
using qw38::gdn::test::kGdnKeyHeads;
using qw38::gdn::test::kGdnQkFp32Abs;
using qw38::gdn::test::kGdnQkFp32Rel;
using qw38::gdn::test::kGdnRecurMultiOAbs;
using qw38::gdn::test::kGdnRecurMultiORel;
using qw38::gdn::test::kGdnRecurMultiSAbs;
using qw38::gdn::test::kGdnRecurMultiSRel;
using qw38::gdn::test::kGdnRecurOAbs;
using qw38::gdn::test::kGdnRecurORel;
using qw38::gdn::test::kGdnRecurSAbs;
using qw38::gdn::test::kGdnRecurSRel;
using qw38::gdn::test::kGdnSElemsPerLayer;
using qw38::gdn::test::kGdnValueHeads;
using qw38::gdn::test::kGdnZWidth;
using qw38::gdn::test::kHidden;
using qw38::gdn::test::kQkvWidth;
using qw38::gdn::test::make_logical;
using qw38::gdn::test::make_view;
using qw38::gdn::test::mixer_views;
using qw38::gdn::test::pack_bf16_tile;
using qw38::gdn::test::pack_q4_from_logical;
using qw38::gdn::test::pattern_f;
using qw38::gdn::test::pattern_h;
using qw38::gdn::test::residual_vec;
using qw38::gdn::test::upload_host_mixer;
using qw38::gdn::test::upload_vec;
using qw38::gdn::test::v_from_convolved;
using qw38::gdn::test::zeros_f;
using qw38::gdn::test::zeros_h;
using qw38::reference::gdn_conv_history_step;
using qw38::reference::gdn_front_reference;
using qw38::reference::gdn_mixer_reference;
using qw38::reference::gdn_prepare;
using qw38::reference::gdn_recurrence_step;
using qw38::runtime::GdnFrontBindViews;
using qw38::runtime::GdnRecurrenceBindViews;
using qw38::runtime::bind_gdn_front_plan;
using qw38::runtime::bind_gdn_plan;
using qw38::runtime::bind_gdn_recurrence_plan;
using qw38::runtime::execute_decode_gdn;
using qw38::runtime::execute_gdn_front;
using qw38::runtime::execute_gdn_recurrence;
using qw38::runtime::kGdnOffO;
using qw38::runtime::kGdnOffU;
using qw38::runtime::kGdnWorkspaceBytesPerToken;

namespace {

void test_multi_step_kernels(Stream const& stream) {
  auto taps = pattern_h(kConvKernel * kQkvWidth, 0.35f);
  auto history = zeros_h(kConvHistoryTaps * kQkvWidth);
  auto d_t = upload_vec(taps, stream);
  auto d_h = upload_vec(history, stream);
  auto d_c = DeviceBuffer::allocate(kQkvWidth * 2);
  if (!d_t || !d_h || !d_c) {
    fail("multi-step alloc");
    return;
  }
  std::uint32_t cpu_cursor = 0;
  std::uint32_t gpu_cursor = 0;
  auto cpu_hist = history;
  for (int step = 0; step < 7; ++step) {
    auto qkv = pattern_h(kQkvWidth, 0.2f * static_cast<float>(step + 1));
    auto cpu_conv = zeros_h(kQkvWidth);
    auto rst = gdn_conv_history_step(qkv, taps, cpu_hist, cpu_cursor, cpu_conv);
    expect(static_cast<bool>(rst), "cpu multi conv");
    auto d_q = upload_vec(qkv, stream);
    if (!d_q) {
      fail("multi qkv upload");
      return;
    }
    auto st = launch_gdn_conv_silu(static_cast<std::uint16_t const*>(d_q->data()),
                                   static_cast<std::uint16_t const*>(d_t->data()),
                                   static_cast<std::uint16_t*>(d_h->data()),
                                   gpu_cursor, static_cast<std::uint16_t*>(d_c->data()),
                                   stream);
    expect(static_cast<bool>(st), "gpu multi conv");
    gpu_cursor = (gpu_cursor + 1u) % 3u;
    auto got = download_vec<std::uint16_t>(*d_c, kQkvWidth, stream);
    if (!got) {
      fail("multi conv download");
      return;
    }
    expect_bf16_close(*got, cpu_conv,
                      std::string("conv step ") + std::to_string(step),
                      kGdnConvBf16Abs, 0.0f);

    std::vector<float> a(kGdnValueHeads, 0.1f * static_cast<float>(step));
    std::vector<float> b(kGdnValueHeads, -0.2f + 0.05f * static_cast<float>(step));
    auto alog = pattern_h(kGdnValueHeads, -0.4f);
    auto dt = pattern_h(kGdnValueHeads, 0.3f);
    std::size_t const nqk = static_cast<std::size_t>(kGdnKeyHeads) * kGdnHeadDim;
    std::vector<float> q_ref(nqk);
    std::vector<float> k_ref(nqk);
    std::vector<float> alpha_ref(kGdnValueHeads);
    std::vector<float> beta_ref(kGdnValueHeads);
    auto pst = gdn_prepare(cpu_conv, a, b, alog, dt, kDefaultRmsEps, q_ref, k_ref,
                           alpha_ref, beta_ref);
    expect(static_cast<bool>(pst), "cpu prepare");
    auto d_a = upload_vec(a, stream);
    auto d_b = upload_vec(b, stream);
    auto d_al = upload_vec(alog, stream);
    auto d_dt = upload_vec(dt, stream);
    auto d_qhat = DeviceBuffer::allocate(nqk * 4);
    auto d_khat = DeviceBuffer::allocate(nqk * 4);
    auto d_alpha = DeviceBuffer::allocate(kGdnValueHeads * 4);
    auto d_beta = DeviceBuffer::allocate(kGdnValueHeads * 4);
    if (!d_a || !d_b || !d_al || !d_dt || !d_qhat || !d_khat || !d_alpha ||
        !d_beta) {
      fail("prepare alloc");
      return;
    }
    st = launch_gdn_prepare(static_cast<std::uint16_t const*>(d_c->data()),
                            static_cast<float const*>(d_a->data()),
                            static_cast<float const*>(d_b->data()),
                            static_cast<std::uint16_t const*>(d_al->data()),
                            static_cast<std::uint16_t const*>(d_dt->data()),
                            kDefaultRmsEps, static_cast<float*>(d_qhat->data()),
                            static_cast<float*>(d_khat->data()),
                            static_cast<float*>(d_alpha->data()),
                            static_cast<float*>(d_beta->data()), stream);
    expect(static_cast<bool>(st), "gpu prepare");
    auto gq = download_vec<float>(*d_qhat, nqk, stream);
    auto gk = download_vec<float>(*d_khat, nqk, stream);
    auto ga = download_vec<float>(*d_alpha, kGdnValueHeads, stream);
    auto gb = download_vec<float>(*d_beta, kGdnValueHeads, stream);
    if (!gq || !gk || !ga || !gb) {
      fail("prepare download");
      return;
    }
    expect_fp32_close(*gq, q_ref, std::string("q step ") + std::to_string(step),
                      kGdnQkFp32Abs, kGdnQkFp32Rel);
    expect_fp32_close(*gk, k_ref, std::string("k step ") + std::to_string(step),
                      kGdnQkFp32Abs, kGdnQkFp32Rel);
    expect_fp32_close(*ga, alpha_ref,
                      std::string("alpha step ") + std::to_string(step),
                      kGdnGateFp32Abs, kGdnGateFp32Rel);
    expect_fp32_close(*gb, beta_ref,
                      std::string("beta step ") + std::to_string(step),
                      kGdnGateFp32Abs, kGdnGateFp32Rel);
  }
  expect(cpu_cursor == gpu_cursor, "cursors stay aligned");
  auto got_h = download_vec<std::uint16_t>(*d_h, kConvHistoryTaps * kQkvWidth, stream);
  if (got_h) {
    expect_bf16_close(*got_h, cpu_hist, "history after 7 steps", 0.0f, 0.0f);
  }
}

struct HostFront {
  PackedMatrix qkv;
  PackedMatrix z;
  PackedMatrix a;
  PackedMatrix b;
  std::vector<std::uint16_t> w_qkv;
  std::vector<std::uint16_t> w_z;
  std::vector<std::uint16_t> w_a;
  std::vector<std::uint16_t> w_b;
  std::vector<std::uint16_t> gamma;
  std::vector<std::uint16_t> taps;
  std::vector<std::uint16_t> a_log;
  std::vector<std::uint16_t> dt_bias;
  std::vector<float> residual;
};

bool make_host(HostFront& host) {
  auto lq = make_logical(LogicalQuantizerId::Q4G64V0, kQkvWidth, kHidden, 1, 0x3C00);
  auto lz = make_logical(LogicalQuantizerId::Q4G64V0, kGdnZWidth, kHidden, -2, 0x3C00);
  fill_logical_pattern(lq, 2);
  fill_logical_pattern(lz, 5);
  if (!pack_q4_from_logical(lq, host.qkv, "qkv") ||
      !pack_q4_from_logical(lz, host.z, "z")) {
    return false;
  }
  auto wq = qw38::compiler::dequantize_to_bf16(lq);
  auto wz = qw38::compiler::dequantize_to_bf16(lz);
  if (!wq || !wz) {
    fail("dequant qkv/z");
    return false;
  }
  host.w_qkv = std::move(*wq);
  host.w_z = std::move(*wz);
  host.w_a = pattern_h(kGdnValueHeads * kHidden, 0.04f);
  host.w_b = pattern_h(kGdnValueHeads * kHidden, -0.03f);
  if (!pack_bf16_tile(host.w_a, kGdnValueHeads, kHidden, host.a, "a") ||
      !pack_bf16_tile(host.w_b, kGdnValueHeads, kHidden, host.b, "b")) {
    return false;
  }
  host.gamma = pattern_h(kHidden, 0.2f);
  host.taps = pattern_h(kConvKernel * kQkvWidth, 0.15f);
  host.a_log = pattern_h(kGdnValueHeads, -0.5f);
  host.dt_bias = pattern_h(kGdnValueHeads, 0.4f);
  host.residual = residual_vec(kHidden, 0.8f);
  return true;
}

void test_front_vs_reference(Stream const& stream) {
  HostFront host;
  if (!make_host(host)) {
    return;
  }
  auto history = zeros_h(kConvHistoryTaps * kQkvWidth);
  std::uint32_t cursor = 0;
  auto cpu = gdn_front_reference(host.residual, host.gamma, kDefaultRmsEps,
                                 host.w_qkv, host.w_z, host.w_a, host.w_b,
                                 host.taps, host.a_log, host.dt_bias, history,
                                 cursor);
  expect(static_cast<bool>(cpu), "cpu front");
  if (!cpu) {
    return;
  }

  auto d_qkv = upload_vec(host.qkv.codes, stream);
  auto d_qkv_s = upload_vec(host.qkv.scales, stream);
  auto d_z = upload_vec(host.z.codes, stream);
  auto d_z_s = upload_vec(host.z.scales, stream);
  auto d_a = upload_vec(host.a.codes, stream);
  auto d_b = upload_vec(host.b.codes, stream);
  auto d_gamma = upload_vec(host.gamma, stream);
  auto d_taps = upload_vec(host.taps, stream);
  auto d_alog = upload_vec(host.a_log, stream);
  auto d_dt = upload_vec(host.dt_bias, stream);
  auto d_res = upload_vec(host.residual, stream);
  auto d_norm = DeviceBuffer::allocate(kHidden * 2);
  auto d_ws = DeviceBuffer::allocate(kGdnWorkspaceBytesPerToken);
  auto d_hist = upload_vec(history, stream);
  if (!d_qkv || !d_qkv_s || !d_z || !d_z_s || !d_a || !d_b || !d_gamma ||
      !d_taps || !d_alog || !d_dt || !d_res || !d_norm || !d_ws || !d_hist) {
    fail("front upload");
    return;
  }
  std::uint32_t gpu_cursor = 0;
  GdnFrontBindViews views;
  views.qkv = make_view(d_qkv->data(), qw38::format::ArithmeticDtype::Bf16,
                        PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped,
                        false, 2, kQkvWidth, kHidden);
  views.qkv_scales = make_view(d_qkv_s->data(), qw38::format::ArithmeticDtype::Fp16,
                               PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped,
                               false, 1, host.qkv.scales.size() / 2);
  views.z = make_view(d_z->data(), qw38::format::ArithmeticDtype::Bf16,
                      PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped, false,
                      2, kGdnZWidth, kHidden);
  views.z_scales = make_view(d_z_s->data(), qw38::format::ArithmeticDtype::Fp16,
                             PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped,
                             false, 1, host.z.scales.size() / 2);
  views.a = make_view(d_a->data(), qw38::format::ArithmeticDtype::Bf16,
                      PhysicalLayoutId::CudaBf16DenseTileV0, StorageClass::Bf16,
                      false, 2, kGdnValueHeads, kHidden);
  views.b = make_view(d_b->data(), qw38::format::ArithmeticDtype::Bf16,
                      PhysicalLayoutId::CudaBf16DenseTileV0, StorageClass::Bf16,
                      false, 2, kGdnValueHeads, kHidden);
  views.gamma = make_view(d_gamma->data(), qw38::format::ArithmeticDtype::Bf16,
                          PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16,
                          false, 1, kHidden);
  views.taps = make_view(d_taps->data(), qw38::format::ArithmeticDtype::Bf16,
                         PhysicalLayoutId::CudaBf16TapMajorV0, StorageClass::Bf16,
                         false, 2, kConvKernel, kQkvWidth);
  views.a_log = make_view(d_alog->data(), qw38::format::ArithmeticDtype::Bf16,
                          PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16,
                          false, 1, kGdnValueHeads);
  views.dt_bias = make_view(d_dt->data(), qw38::format::ArithmeticDtype::Bf16,
                            PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16,
                            false, 1, kGdnValueHeads);
  views.residual = make_view(d_res->data(), qw38::format::ArithmeticDtype::Fp32,
                             PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32,
                             true, 1, kHidden);
  views.normalized = make_view(d_norm->data(), qw38::format::ArithmeticDtype::Bf16,
                               PhysicalLayoutId::CudaBf16RowMajorV0,
                               StorageClass::Bf16, true, 1, kHidden);
  views.workspace = make_view(d_ws->data(), qw38::format::ArithmeticDtype::Fp32,
                              PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32,
                              true, 1, kGdnWorkspaceBytesPerToken / 4);
  views.history = make_view(d_hist->data(), qw38::format::ArithmeticDtype::Bf16,
                            PhysicalLayoutId::CudaBf16ConvHistoryV0,
                            StorageClass::Bf16, true, 2, kConvHistoryTaps, kQkvWidth);
  views.host_cursor = &gpu_cursor;
  views.language_layer = 0;
  auto plan = bind_gdn_front_plan(views, stream);
  expect(static_cast<bool>(plan), "front bind");
  if (!plan) {
    return;
  }
  auto st = execute_gdn_front(*plan);
  expect(static_cast<bool>(st), "front execute");
  if (!st) {
    fail(qw38::runtime::error_message(st.error()));
    return;
  }
  expect(gpu_cursor == cpu->cursor, "front cursor");

  auto* ws = static_cast<std::byte*>(d_ws->data());
  auto copy_ws = [&](std::uint64_t off, std::uint64_t bytes, auto* dst) {
    auto stc = qw38::cuda::copy_d2h(dst, ws + off, bytes, stream);
    return static_cast<bool>(stc) && static_cast<bool>(stream.sync());
  };
  std::vector<std::uint16_t> g_qkv(kQkvWidth);
  std::vector<std::uint16_t> g_z(kGdnZWidth);
  std::vector<std::uint16_t> g_conv(kQkvWidth);
  std::size_t const nqk = static_cast<std::size_t>(kGdnKeyHeads) * kGdnHeadDim;
  std::vector<float> g_q(nqk);
  std::vector<float> g_k(nqk);
  std::vector<float> g_a(kGdnValueHeads);
  std::vector<float> g_b(kGdnValueHeads);
  std::vector<float> g_alpha(kGdnValueHeads);
  std::vector<float> g_beta(kGdnValueHeads);
  std::vector<std::uint16_t> g_norm(kHidden);
  std::vector<std::uint16_t> g_hist(kConvHistoryTaps * kQkvWidth);
  if (!copy_ws(qw38::runtime::kGdnOffQkv, kQkvWidth * 2, g_qkv.data()) ||
      !copy_ws(qw38::runtime::kGdnOffZ, kGdnZWidth * 2, g_z.data()) ||
      !copy_ws(qw38::runtime::kGdnOffConvolved, kQkvWidth * 2, g_conv.data()) ||
      !copy_ws(qw38::runtime::kGdnOffQHat, nqk * 4, g_q.data()) ||
      !copy_ws(qw38::runtime::kGdnOffKHat, nqk * 4, g_k.data()) ||
      !copy_ws(qw38::runtime::kGdnOffA, kGdnValueHeads * 4, g_a.data()) ||
      !copy_ws(qw38::runtime::kGdnOffB, kGdnValueHeads * 4, g_b.data()) ||
      !copy_ws(qw38::runtime::kGdnOffAlpha, kGdnValueHeads * 4, g_alpha.data()) ||
      !copy_ws(qw38::runtime::kGdnOffBeta, kGdnValueHeads * 4, g_beta.data())) {
    fail("workspace download");
    return;
  }
  auto gn = download_vec<std::uint16_t>(*d_norm, kHidden, stream);
  auto gh = download_vec<std::uint16_t>(*d_hist, kConvHistoryTaps * kQkvWidth, stream);
  if (!gn || !gh) {
    fail("norm/hist download");
    return;
  }
  expect_bf16_close(*gn, cpu->normalized, "stage rms",
                    qw38::reference::tol::kRmsBf16Abs, 0.0f);
  expect_bf16_close(g_qkv, cpu->qkv, "stage qkv",
                    qw38::cuda::decode_mmv_tol::kBf16StoreAbs, 1.0e-4f);
  expect_bf16_close(g_z, cpu->z, "stage z",
                    qw38::cuda::decode_mmv_tol::kBf16StoreAbs, 1.0e-4f);
  expect_fp32_close(g_a, cpu->a, "stage a", qw38::cuda::decode_mmv_tol::kFp32Abs,
                    qw38::cuda::decode_mmv_tol::kFp32Rel);
  expect_fp32_close(g_b, cpu->b, "stage b", qw38::cuda::decode_mmv_tol::kFp32Abs,
                    qw38::cuda::decode_mmv_tol::kFp32Rel);
  expect_bf16_close(g_conv, cpu->convolved, "stage conv", kGdnConvBf16Abs, 1.0e-4f);
  expect_fp32_close(g_q, cpu->q_hat, "stage q_hat", kGdnQkFp32Abs, kGdnQkFp32Rel);
  expect_fp32_close(g_k, cpu->k_hat, "stage k_hat", kGdnQkFp32Abs, kGdnQkFp32Rel);
  expect_fp32_close(g_alpha, cpu->alpha, "stage alpha", kGdnGateFp32Abs,
                    kGdnGateFp32Rel);
  expect_fp32_close(g_beta, cpu->beta, "stage beta", kGdnGateFp32Abs,
                    kGdnGateFp32Rel);
  expect_bf16_close(*gh, cpu->history, "stage history", 0.0f, 0.0f);

  auto v_cpu = v_from_convolved(cpu->convolved);
  auto s_cpu = zeros_f(static_cast<std::uint32_t>(kGdnSElemsPerLayer));
  auto o_cpu = zeros_f(kGdnValueHeads * kGdnHeadDim);
  auto rst = gdn_recurrence_step(cpu->q_hat, cpu->k_hat, cpu->alpha, cpu->beta, v_cpu,
                                 s_cpu, o_cpu);
  expect(static_cast<bool>(rst), "cpu recurrence after front");
  auto d_s = DeviceBuffer::allocate(kGdnSElemsPerLayer * 4);
  if (!d_s) {
    fail("S alloc");
    return;
  }
  expect(static_cast<bool>(qw38::cuda::zero(*d_s, stream)), "zero S");
  GdnRecurrenceBindViews rv;
  rv.q_hat = plan->scratch.q_hat;
  rv.k_hat = plan->scratch.k_hat;
  rv.alpha = plan->scratch.alpha;
  rv.beta = plan->scratch.beta;
  rv.v = plan->scratch.v;
  rv.s = make_view(d_s->data(), qw38::format::ArithmeticDtype::Fp32,
                   PhysicalLayoutId::CudaFp32GdnSHvKV0, StorageClass::Fp32, true, 1,
                   kGdnSElemsPerLayer);
  rv.o = plan->scratch.o;
  rv.s_layer = 0;
  rv.language_layer = 0;
  auto rplan = bind_gdn_recurrence_plan(rv, stream);
  expect(static_cast<bool>(rplan), "bind recurrence after front");
  if (!rplan) {
    return;
  }
  auto est = execute_gdn_recurrence(*rplan);
  expect(static_cast<bool>(est), "execute recurrence after front");
  auto g_s = download_vec<float>(*d_s, kGdnSElemsPerLayer, stream);
  std::vector<float> g_o(static_cast<std::size_t>(kGdnValueHeads) * kGdnHeadDim);
  if (!g_s || !copy_ws(kGdnOffO, g_o.size() * 4, g_o.data())) {
    fail("recurrence download");
    return;
  }
  expect_fp32_close(*g_s, s_cpu, "front+recur S", kGdnRecurSAbs, kGdnRecurSRel);
  expect_fp32_close(g_o, o_cpu, "front+recur o", kGdnRecurOAbs, kGdnRecurORel);
}

void l2_normalize_heads(std::vector<float>& x) {
  for (std::uint32_t h = 0; h < kGdnKeyHeads; ++h) {
    float sumsq = 0.0f;
    for (std::uint32_t i = 0; i < kGdnHeadDim; ++i) {
      float const v = x[static_cast<std::size_t>(h) * kGdnHeadDim + i];
      sumsq += v * v;
    }
    float const inv = 1.0f / std::sqrt(sumsq + 1.0e-6f);
    for (std::uint32_t i = 0; i < kGdnHeadDim; ++i) {
      x[static_cast<std::size_t>(h) * kGdnHeadDim + i] *= inv;
    }
  }
}

void fill_step_inputs(int step, std::vector<float>& q, std::vector<float>& k,
                      std::vector<float>& alpha, std::vector<float>& beta,
                      std::vector<std::uint16_t>& v) {
  float const s = 0.15f + 0.01f * static_cast<float>(step);
  q = pattern_f(static_cast<std::uint32_t>(q.size()), s);
  k = pattern_f(static_cast<std::uint32_t>(k.size()), -s);
  l2_normalize_heads(q);
  l2_normalize_heads(k);
  alpha.assign(kGdnValueHeads, 0.35f + 0.002f * static_cast<float>(step % 17));
  beta.assign(kGdnValueHeads, 0.45f + 0.003f * static_cast<float>(step % 11));
  v = pattern_h(static_cast<std::uint32_t>(v.size()),
                0.25f + 0.02f * static_cast<float>(step % 9));
}

bool run_gpu_step(Stream const& stream, std::vector<float> const& q,
                  std::vector<float> const& k, std::vector<float> const& alpha,
                  std::vector<float> const& beta,
                  std::vector<std::uint16_t> const& v, DeviceBuffer& d_s,
                  DeviceBuffer& d_o) {
  auto d_q = upload_vec(q, stream);
  auto d_k = upload_vec(k, stream);
  auto d_a = upload_vec(alpha, stream);
  auto d_b = upload_vec(beta, stream);
  auto d_v = upload_vec(v, stream);
  if (!d_q || !d_k || !d_a || !d_b || !d_v) {
    fail("step upload");
    return false;
  }
  auto st = launch_gdn_recurrence(static_cast<float const*>(d_q->data()),
                                  static_cast<float const*>(d_k->data()),
                                  static_cast<float const*>(d_a->data()),
                                  static_cast<float const*>(d_b->data()),
                                  static_cast<std::uint16_t const*>(d_v->data()),
                                  static_cast<float*>(d_s.data()), 0,
                                  static_cast<float*>(d_o.data()), stream);
  expect(static_cast<bool>(st), "recurrence launch");
  return static_cast<bool>(st);
}

void test_recurrence_steps_and_adversarial(Stream const& stream) {
  std::size_t const nqk = static_cast<std::size_t>(kGdnKeyHeads) * kGdnHeadDim;
  std::size_t const nv = static_cast<std::size_t>(kGdnValueHeads) * kGdnHeadDim;
  std::size_t const ns = static_cast<std::size_t>(kGdnSElemsPerLayer);

  auto d_s = DeviceBuffer::allocate(ns * 4);
  auto d_o = DeviceBuffer::allocate(nv * 4);
  if (!d_s || !d_o) {
    fail("recur buffers");
    return;
  }
  expect(static_cast<bool>(qw38::cuda::zero(*d_s, stream)), "zero S");

  std::vector<float> q(nqk), k(nqk), alpha(kGdnValueHeads), beta(kGdnValueHeads);
  std::vector<std::uint16_t> v(nv);
  auto s_cpu = zeros_f(static_cast<std::uint32_t>(ns));
  auto o_cpu = zeros_f(static_cast<std::uint32_t>(nv));
  int const checkpoints[] = {1, 2, 17, 128};
  int next = 0;
  for (int step = 0; step < 128; ++step) {
    fill_step_inputs(step, q, k, alpha, beta, v);
    auto rst = gdn_recurrence_step(q, k, alpha, beta, v, s_cpu, o_cpu);
    expect(static_cast<bool>(rst), "cpu step");
    if (!run_gpu_step(stream, q, k, alpha, beta, v, *d_s, *d_o)) {
      return;
    }
    if (next < 4 && step + 1 == checkpoints[next]) {
      float const sa = (step + 1 <= 2) ? kGdnRecurSAbs : kGdnRecurMultiSAbs;
      float const sr = (step + 1 <= 2) ? kGdnRecurSRel : kGdnRecurMultiSRel;
      float const oa = (step + 1 <= 2) ? kGdnRecurOAbs : kGdnRecurMultiOAbs;
      float const orr = (step + 1 <= 2) ? kGdnRecurORel : kGdnRecurMultiORel;
      auto got_s = download_vec<float>(*d_s, ns, stream);
      auto got_o = download_vec<float>(*d_o, nv, stream);
      if (!got_s || !got_o) {
        fail("checkpoint download");
        return;
      }
      expect_fp32_close(*got_s, s_cpu,
                        std::string("S steps=") + std::to_string(step + 1), sa, sr);
      expect_fp32_close(*got_o, o_cpu,
                        std::string("o steps=") + std::to_string(step + 1), oa, orr);
      ++next;
    }
  }

  struct Adv {
    char const* name;
    float a;
    float b;
    float qv;
    float kv;
    float vv;
  };
  Adv cases[] = {
      {"alpha0", 0.0f, 1.0f, 0.3f, 0.4f, 0.5f},
      {"beta0", 1.0f, 0.0f, 0.3f, 0.4f, 0.5f},
      {"alpha1-beta1", 1.0f, 1.0f, 0.2f, -0.2f, 0.7f},
      {"tiny-alpha", 1.0e-8f, 0.9f, 0.5f, 0.5f, 0.5f},
      {"tiny-beta", 0.9f, 1.0e-8f, 0.5f, 0.5f, 0.5f},
      {"large-v", 0.7f, 0.8f, 0.1f, 0.2f, 50.0f},
  };
  for (auto const& c : cases) {
    q = pattern_f(static_cast<std::uint32_t>(nqk), c.qv);
    k = pattern_f(static_cast<std::uint32_t>(nqk), c.kv);
    l2_normalize_heads(q);
    l2_normalize_heads(k);
    alpha.assign(kGdnValueHeads, c.a);
    beta.assign(kGdnValueHeads, c.b);
    v = pattern_h(static_cast<std::uint32_t>(nv), c.vv);
    auto s0 = pattern_f(static_cast<std::uint32_t>(ns), 0.05f);
    auto s_ref = s0;
    auto o_ref = zeros_f(static_cast<std::uint32_t>(nv));
    auto rst = gdn_recurrence_step(q, k, alpha, beta, v, s_ref, o_ref);
    expect(static_cast<bool>(rst), std::string(c.name) + " cpu");
    auto d_si = upload_vec(s0, stream);
    if (!d_si) {
      fail(std::string(c.name) + " S upload");
      return;
    }
    if (!run_gpu_step(stream, q, k, alpha, beta, v, *d_si, *d_o)) {
      return;
    }
    auto got_s = download_vec<float>(*d_si, ns, stream);
    auto got_o = download_vec<float>(*d_o, nv, stream);
    if (!got_s || !got_o) {
      fail(std::string(c.name) + " download");
      return;
    }
    expect_fp32_close(*got_s, s_ref, std::string(c.name) + " S", kGdnRecurSAbs,
                      kGdnRecurSRel);
    expect_fp32_close(*got_o, o_ref, std::string(c.name) + " o", kGdnRecurOAbs,
                      kGdnRecurORel);
  }
}

bool make_q4_mixer_host(HostGdnMixer& host) {
  auto lq = make_logical(LogicalQuantizerId::Q4G64V0, kQkvWidth, kHidden, 2, 0x3C00);
  auto lz = make_logical(LogicalQuantizerId::Q4G64V0, kGdnZWidth, kHidden, -1, 0x3C00);
  auto lo = make_logical(LogicalQuantizerId::Q4G64V0, kHidden, kGdnZWidth, 3, 0x3C00);
  fill_logical_pattern(lq, 3);
  fill_logical_pattern(lz, 6);
  fill_logical_pattern(lo, 10);
  if (!pack_q4_from_logical(lq, host.qkv, "mix qkv") ||
      !pack_q4_from_logical(lz, host.z, "mix z") ||
      !pack_q4_from_logical(lo, host.out, "mix out")) {
    return false;
  }
  auto wq = qw38::compiler::dequantize_to_bf16(lq);
  auto wz = qw38::compiler::dequantize_to_bf16(lz);
  auto wo = qw38::compiler::dequantize_to_bf16(lo);
  if (!wq || !wz || !wo) {
    fail("mix dequant");
    return false;
  }
  host.w_qkv = std::move(*wq);
  host.w_z = std::move(*wz);
  host.w_out = std::move(*wo);
  host.w_a = pattern_h(kGdnValueHeads * kHidden, 0.04f);
  host.w_b = pattern_h(kGdnValueHeads * kHidden, -0.03f);
  if (!pack_bf16_tile(host.w_a, kGdnValueHeads, kHidden, host.a, "mix a") ||
      !pack_bf16_tile(host.w_b, kGdnValueHeads, kHidden, host.b, "mix b")) {
    return false;
  }
  host.gamma = pattern_h(kHidden, 0.18f);
  host.gated_gamma = pattern_h(kGdnHeadDim, 0.65f);
  host.taps = pattern_h(kConvKernel * kQkvWidth, 0.14f);
  host.a_log = pattern_h(kGdnValueHeads, -0.4f);
  host.dt_bias = pattern_h(kGdnValueHeads, 0.35f);
  host.residual = residual_vec(kHidden, 0.77f);
  return true;
}

void expect_eight_launch_schedule(qw38::runtime::GdnPlan const& plan,
                                  Stream const& stream, std::string_view tag) {
  if (auto st = stream.sync(); !st) {
    fail(std::string(tag) + " launch-count pre-sync");
    return;
  }
  std::uint32_t const saved_cursor = *plan.front.host_cursor;
  cudaGraph_t graph = nullptr;
  auto status = cudaStreamBeginCapture(stream.native(), cudaStreamCaptureModeThreadLocal);
  if (status != cudaSuccess) {
    fail(std::string(tag) + " begin launch-count capture: " +
         cudaGetErrorString(status));
    return;
  }
  auto run = execute_decode_gdn(plan);
  status = cudaStreamEndCapture(stream.native(), &graph);
  *plan.front.host_cursor = saved_cursor;
  if (!run || status != cudaSuccess || graph == nullptr) {
    if (graph != nullptr) {
      (void)cudaGraphDestroy(graph);
    }
    fail(std::string(tag) + " capture eight-launch schedule");
    return;
  }

  std::size_t node_count = 0;
  status = cudaGraphGetNodes(graph, nullptr, &node_count);
  if (status != cudaSuccess) {
    (void)cudaGraphDestroy(graph);
    fail(std::string(tag) + " count captured nodes");
    return;
  }
  std::vector<cudaGraphNode_t> nodes(node_count);
  status = cudaGraphGetNodes(graph, nodes.data(), &node_count);
  std::size_t kernel_count = 0;
  std::size_t memcpy_count = 0;
  if (status == cudaSuccess) {
    for (auto node : nodes) {
      cudaGraphNodeType type{};
      if (cudaGraphNodeGetType(node, &type) != cudaSuccess) {
        status = cudaErrorUnknown;
        break;
      }
      kernel_count += type == cudaGraphNodeTypeKernel ? 1u : 0u;
      memcpy_count += type == cudaGraphNodeTypeMemcpy ? 1u : 0u;
    }
  }
  (void)cudaGraphDestroy(graph);
  expect(status == cudaSuccess, std::string(tag) + " inspect captured nodes");
  expect(node_count == qw38::runtime::kGdnMixerRegions,
         std::string(tag) + " captures exactly eight schedule nodes");
  expect(kernel_count == qw38::runtime::kGdnMixerRegions,
         std::string(tag) + " captures exactly eight kernel launches");
  expect(memcpy_count == 0,
         std::string(tag) + " output-residual region has no copy node");
}

bool compare_mixer(HostGdnMixer const& host, DeviceGdnMixer& dev, Stream const& stream,
                   std::string_view tag, int tokens) {
  auto views = mixer_views(dev);
  auto plan = bind_gdn_plan(views, stream);
  if (!plan) {
    fail(std::string(tag) + " bind: " + qw38::runtime::error_message(plan.error()));
    return false;
  }
  expect_eight_launch_schedule(*plan, stream, tag);
  std::vector<std::uint16_t> hist(kConvHistoryTaps * kQkvWidth,
                                  qw38::format::fp32_to_bf16_rne(0.0f));
  std::uint32_t cursor = 0;
  auto s_cpu = zeros_f(static_cast<std::uint32_t>(kGdnSElemsPerLayer));
  auto residual = host.residual;
  for (int t = 0; t < tokens; ++t) {
    if (t > 0) {
      residual = residual_vec(kHidden, 0.5f + 0.07f * static_cast<float>(t));
      auto up = qw38::cuda::copy_h2d(dev.residual.data(), residual.data(),
                                     residual.size() * 4u, stream);
      if (!up) {
        fail(std::string(tag) + " residual upload");
        return false;
      }
    }
    auto cpu = gdn_mixer_reference(residual, host.gamma, host.gated_gamma,
                                   kDefaultRmsEps, host.w_qkv, host.w_z, host.w_a,
                                   host.w_b, host.w_out, host.taps, host.a_log,
                                   host.dt_bias, hist, cursor, s_cpu);
    if (!cpu) {
      fail(std::string(tag) + " cpu: " +
           qw38::reference::error_message(cpu.error()));
      return false;
    }
    hist = cpu->history;
    cursor = cpu->cursor;
    s_cpu = cpu->s;

    auto st = execute_decode_gdn(*plan);
    if (!st) {
      fail(std::string(tag) + " execute: " +
           qw38::runtime::error_message(st.error()));
      return false;
    }

    auto* ws = static_cast<std::byte*>(dev.workspace.data());
    auto copy_ws = [&](std::uint64_t off, std::uint64_t bytes, auto* dst) {
      auto stc = qw38::cuda::copy_d2h(dst, ws + off, bytes, stream);
      return static_cast<bool>(stc) && static_cast<bool>(stream.sync());
    };
    std::vector<std::uint16_t> g_qkv(kQkvWidth);
    std::vector<std::uint16_t> g_z(kGdnZWidth);
    std::vector<std::uint16_t> g_conv(kQkvWidth);
    std::vector<std::uint16_t> g_u(kGdnValueHeads * kGdnHeadDim);
    std::vector<float> g_o(static_cast<std::size_t>(kGdnValueHeads) * kGdnHeadDim);
    std::vector<float> g_q(static_cast<std::size_t>(kGdnKeyHeads) * kGdnHeadDim);
    std::vector<float> g_alpha(kGdnValueHeads);
    if (!copy_ws(qw38::runtime::kGdnOffQkv, kQkvWidth * 2, g_qkv.data()) ||
        !copy_ws(qw38::runtime::kGdnOffZ, kGdnZWidth * 2, g_z.data()) ||
        !copy_ws(qw38::runtime::kGdnOffConvolved, kQkvWidth * 2, g_conv.data()) ||
        !copy_ws(qw38::runtime::kGdnOffQHat, g_q.size() * 4, g_q.data()) ||
        !copy_ws(qw38::runtime::kGdnOffAlpha, kGdnValueHeads * 4, g_alpha.data()) ||
        !copy_ws(kGdnOffO, g_o.size() * 4, g_o.data()) ||
        !copy_ws(kGdnOffU, g_u.size() * 2, g_u.data())) {
      fail(std::string(tag) + " workspace download");
      return false;
    }
    auto g_norm = download_vec<std::uint16_t>(dev.normalized, kHidden, stream);
    auto g_hist = download_vec<std::uint16_t>(dev.history, kConvHistoryTaps * kQkvWidth,
                                             stream);
    auto g_s = download_vec<float>(dev.s, kGdnSElemsPerLayer, stream);
    auto g_h = download_vec<float>(dev.residual, kHidden, stream);
    auto g_hmid = download_vec<float>(dev.residual_out, kHidden, stream);
    if (!g_norm || !g_hist || !g_s || !g_h || !g_hmid) {
      fail(std::string(tag) + " download");
      return false;
    }
    std::string step = std::string(tag) + " t" + std::to_string(t);
    expect_fp32_close(*g_h, residual, step + " residual live", 0.0f, 0.0f);
    expect_bf16_close(*g_norm, cpu->front.normalized, step + " rms",
                      qw38::reference::tol::kRmsBf16Abs, 0.0f);
    expect_bf16_close(g_qkv, cpu->front.qkv, step + " qkv",
                      qw38::cuda::decode_mmv_tol::kBf16StoreAbs, 1.0e-4f);
    expect_bf16_close(g_z, cpu->front.z, step + " z",
                      qw38::cuda::decode_mmv_tol::kBf16StoreAbs, 1.0e-4f);
    expect_bf16_close(g_conv, cpu->front.convolved, step + " conv", kGdnConvBf16Abs,
                      1.0e-4f);
    expect_fp32_close(g_q, cpu->front.q_hat, step + " q_hat", kGdnQkFp32Abs,
                      kGdnQkFp32Rel);
    expect_fp32_close(g_alpha, cpu->front.alpha, step + " alpha", kGdnGateFp32Abs,
                      kGdnGateFp32Rel);
    float const sa = (t == 0) ? kGdnRecurSAbs : kGdnRecurMultiSAbs;
    float const sr = (t == 0) ? kGdnRecurSRel : kGdnRecurMultiSRel;
    float const oa = (t == 0) ? kGdnRecurOAbs : kGdnRecurMultiOAbs;
    float const orr = (t == 0) ? kGdnRecurORel : kGdnRecurMultiORel;
    expect_fp32_close(*g_s, cpu->s, step + " S", sa, sr);
    expect_fp32_close(g_o, cpu->o, step + " o", oa, orr);
    expect_bf16_close(g_u, cpu->u, step + " u", qw38::reference::tol::kGdnUAbs,
                      qw38::reference::tol::kGdnURel);
    expect_bf16_close(*g_hist, cpu->history, step + " history", 0.0f, 0.0f);
    expect_fp32_close(*g_hmid, cpu->residual, step + " h_mid",
                      qw38::reference::tol::kGdnMixerResidualAbs,
                      qw38::reference::tol::kGdnMixerResidualRel);
    expect(dev.cursor == cpu->cursor, step + " cursor");
  }
  return true;
}

void test_mixer_q4_and_bf16(Stream const& stream) {
  HostGdnMixer host;
  if (!make_q4_mixer_host(host)) {
    return;
  }
  DeviceGdnMixer dev;
  if (!upload_host_mixer(host, dev, stream)) {
    return;
  }
  (void)compare_mixer(host, dev, stream, "q4-mixer", 3);

  HostGdnMixer bf;
  auto src_q = qw38::mlp::test::bf16_matrix(kQkvWidth, kHidden, 0.12f, 1);
  auto src_z = qw38::mlp::test::bf16_matrix(kGdnZWidth, kHidden, 0.11f, 4);
  auto src_o = qw38::mlp::test::bf16_matrix(kHidden, kGdnZWidth, 0.09f, 8);
  auto pq = qw38::format::pack_bf16_dense_tile_v0(src_q, kQkvWidth, kHidden);
  auto pz = qw38::format::pack_bf16_dense_tile_v0(src_z, kGdnZWidth, kHidden);
  auto po = qw38::format::pack_bf16_dense_tile_v0(src_o, kHidden, kGdnZWidth);
  if (!pq || !pz || !po) {
    fail("bf16 mixer pack");
    return;
  }
  bf.qkv = std::move(*pq);
  bf.z = std::move(*pz);
  bf.out = std::move(*po);
  bf.w_qkv = qw38::mlp::test::bytes_to_u16(src_q);
  bf.w_z = qw38::mlp::test::bytes_to_u16(src_z);
  bf.w_out = qw38::mlp::test::bytes_to_u16(src_o);
  bf.w_a = pattern_h(kGdnValueHeads * kHidden, 0.05f);
  bf.w_b = pattern_h(kGdnValueHeads * kHidden, -0.02f);
  if (!pack_bf16_tile(bf.w_a, kGdnValueHeads, kHidden, bf.a, "bf a") ||
      !pack_bf16_tile(bf.w_b, kGdnValueHeads, kHidden, bf.b, "bf b")) {
    return;
  }
  bf.gamma = pattern_h(kHidden, 0.16f);
  bf.gated_gamma = pattern_h(kGdnHeadDim, 0.8f);
  bf.taps = pattern_h(kConvKernel * kQkvWidth, 0.13f);
  bf.a_log = pattern_h(kGdnValueHeads, -0.35f);
  bf.dt_bias = pattern_h(kGdnValueHeads, 0.28f);
  bf.residual = residual_vec(kHidden, 0.6f);
  DeviceGdnMixer bdev;
  if (!upload_host_mixer(bf, bdev, stream)) {
    return;
  }
  (void)compare_mixer(bf, bdev, stream, "bf16-mixer", 2);
}

}  // namespace

int main() {
  auto stream = Stream::create();
  if (!stream) {
    fail("stream");
    return 1;
  }
  test_multi_step_kernels(*stream);
  test_front_vs_reference(*stream);
  test_recurrence_steps_and_adversarial(*stream);
  test_mixer_q4_and_bf16(*stream);
  if (g_failures != 0) {
    std::cerr << g_failures << " gdn reference failures\n";
    return 1;
  }
  std::cout << "gdn reference ok\n";
  return 0;
}
