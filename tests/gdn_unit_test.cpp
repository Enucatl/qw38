#include "gdn_support.hpp"

#include "cuda/alloc.hpp"
#include "cuda/copy.hpp"
#include "runtime/runtime.hpp"
#include "runtime_support.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using qw38::cuda::DeviceBuffer;
using qw38::cuda::Stream;
using qw38::cuda::launch_gdn_conv_silu;
using qw38::cuda::launch_gdn_prepare;
using qw38::cuda::launch_gdn_recurrence;
using qw38::cuda::malloc_count;
using qw38::format::ArithmeticDtype;
using qw38::format::LogicalQuantizerId;
using qw38::format::PhysicalLayoutId;
using qw38::format::StorageClass;
using qw38::gdn::test::expect;
using qw38::gdn::test::expect_bf16_close;
using qw38::gdn::test::expect_fp32_close;
using qw38::gdn::test::fail;
using qw38::gdn::test::fill_logical_pattern;
using qw38::gdn::test::filled_h;
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
using qw38::gdn::test::kGdnRecurOAbs;
using qw38::gdn::test::kGdnRecurORel;
using qw38::gdn::test::kGdnRecurSAbs;
using qw38::gdn::test::kGdnRecurSRel;
using qw38::gdn::test::kGdnSElemsPerLayer;
using qw38::gdn::test::kGdnValueHeads;
using qw38::gdn::test::kGdnZWidth;
using qw38::gdn::test::kHidden;
using qw38::gdn::test::kQkvWidth;
using qw38::gdn::test::make_view;
using qw38::gdn::test::pattern_f;
using qw38::gdn::test::pattern_h;
using qw38::gdn::test::zeros_f;
using qw38::gdn::test::zeros_h;
using qw38::gdn::test::DeviceGdnMixer;
using qw38::gdn::test::HostGdnMixer;
using qw38::gdn::test::download_vec;
using qw38::gdn::test::filled_f;
using qw38::gdn::test::gdn_s_index;
using qw38::gdn::test::gdn_state_view;
using qw38::gdn::test::make_logical;
using qw38::gdn::test::mixer_views;
using qw38::gdn::test::pack_bf16_tile;
using qw38::gdn::test::pack_q4_from_logical;
using qw38::gdn::test::residual_vec;
using qw38::gdn::test::upload_host_mixer;
using qw38::gdn::test::upload_vec;
using qw38::reference::gdn_alpha_beta;
using qw38::reference::gdn_conv_history_step;
using qw38::reference::gdn_key_head;
using qw38::reference::gdn_prepare;
using qw38::reference::gdn_recurrence_step;
using qw38::runtime::GdnBindViews;
using qw38::runtime::GdnFrontBindViews;
using qw38::runtime::GdnRecurrenceBindViews;
using qw38::runtime::bind_gdn_front_plan;
using qw38::runtime::bind_gdn_plan;
using qw38::runtime::bind_gdn_recurrence_plan;
using qw38::runtime::bind_gdn_workspace;
using qw38::runtime::execute_decode_gdn;
using qw38::runtime::execute_gdn_front;
using qw38::runtime::execute_gdn_recurrence;
using qw38::runtime::gdn_state_index;
using qw38::runtime::gdn_s_byte_offset;
using qw38::runtime::kGdnBytesQHat;
using qw38::runtime::kGdnOffO;
using qw38::runtime::kGdnOffQHat;
using qw38::runtime::kGdnWorkspaceBytesPerToken;

namespace {

void* dummy_ptr(std::uintptr_t v) { return reinterpret_cast<void*>(v); }

GdnFrontBindViews dummy_ok_views(std::uint32_t* cursor) {
  GdnFrontBindViews v;
  auto* p = dummy_ptr(0x100000);
  v.qkv = make_view(dummy_ptr(0x110000), ArithmeticDtype::Bf16,
                    PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped, false,
                    2, kQkvWidth, kHidden);
  v.z = make_view(dummy_ptr(0x120000), ArithmeticDtype::Bf16,
                  PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped, false, 2,
                  kGdnZWidth, kHidden);
  std::uint64_t const qkv_scales =
      static_cast<std::uint64_t>(kQkvWidth) * (kHidden / 64u);
  std::uint64_t const z_scales =
      static_cast<std::uint64_t>(kGdnZWidth) * (kHidden / 64u);
  v.qkv_scales = make_view(dummy_ptr(0x130000), ArithmeticDtype::Fp16,
                           PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped,
                           false, 1, qkv_scales);
  v.z_scales = make_view(dummy_ptr(0x140000), ArithmeticDtype::Fp16,
                         PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped,
                         false, 1, z_scales);
  v.a = make_view(dummy_ptr(0x150000), ArithmeticDtype::Bf16,
                  PhysicalLayoutId::CudaBf16DenseTileV0, StorageClass::Bf16, false, 2,
                  kGdnValueHeads, kHidden);
  v.b = make_view(dummy_ptr(0x160000), ArithmeticDtype::Bf16,
                  PhysicalLayoutId::CudaBf16DenseTileV0, StorageClass::Bf16, false, 2,
                  kGdnValueHeads, kHidden);
  v.gamma = make_view(dummy_ptr(0x170000), ArithmeticDtype::Bf16,
                      PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16, false,
                      1, kHidden);
  v.taps = make_view(dummy_ptr(0x180000), ArithmeticDtype::Bf16,
                     PhysicalLayoutId::CudaBf16TapMajorV0, StorageClass::Bf16, false,
                     2, kConvKernel, kQkvWidth);
  v.a_log = make_view(dummy_ptr(0x190000), ArithmeticDtype::Bf16,
                      PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16, false,
                      1, kGdnValueHeads);
  v.dt_bias = make_view(dummy_ptr(0x1A0000), ArithmeticDtype::Bf16,
                        PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16,
                        false, 1, kGdnValueHeads);
  v.residual = make_view(dummy_ptr(0x1B0000), ArithmeticDtype::Fp32,
                         PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, true,
                         1, kHidden);
  v.normalized = make_view(dummy_ptr(0x1C0000), ArithmeticDtype::Bf16,
                           PhysicalLayoutId::CudaBf16RowMajorV0, StorageClass::Bf16,
                           true, 1, kHidden);
  v.workspace = make_view(p, ArithmeticDtype::Fp32,
                          PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32,
                          true, 1, kGdnWorkspaceBytesPerToken / 4);
  v.history = make_view(dummy_ptr(0x1D0000), ArithmeticDtype::Bf16,
                        PhysicalLayoutId::CudaBf16ConvHistoryV0, StorageClass::Bf16,
                        true, 2, kConvHistoryTaps, kQkvWidth);
  auto cursor_slot = qw38::runtime::ConvCursorSlot::bind(cursor);
  if (cursor_slot) {
    v.cursor = *cursor_slot;
  }
  v.language_layer = 0;
  return v;
}

void test_bind_errors(Stream const& stream) {
  std::uint32_t cursor = 0;
  auto views = dummy_ok_views(&cursor);
  auto ok = bind_gdn_front_plan(views, stream);
  expect(static_cast<bool>(ok), "valid dummy bind");
  if (ok) {
    expect(!ok->cursor.commit_advance(1) && cursor == 0,
           "out-of-order cursor commit cannot mutate cursor");
    expect(ok->scratch.q_hat.extent[0] == kGdnKeyHeads, "q_hat is 16 heads");
    expect(ok->scratch.k_hat.extent[0] == kGdnKeyHeads, "k_hat is 16 heads");
    expect(ok->scratch.v.pointer ==
               static_cast<std::byte*>(ok->scratch.convolved.pointer) +
                   4096u * 2u,
           "v aliases convolved[4096:]");
    expect(static_cast<std::byte*>(ok->scratch.q_hat.pointer) -
                   static_cast<std::byte*>(ok->scratch.qkv.pointer) ==
               static_cast<std::ptrdiff_t>(kGdnOffQHat),
           "q_hat scratch offset");
    expect(ok->scratch.q_hat.extent[0] * ok->scratch.q_hat.extent[1] * 4u ==
               kGdnBytesQHat,
           "q_hat bytes");
    expect(static_cast<std::byte*>(ok->scratch.o.pointer) -
                   static_cast<std::byte*>(ok->scratch.qkv.pointer) ==
               static_cast<std::ptrdiff_t>(kGdnOffO),
           "o scratch offset");
    expect(ok->scratch.o.extent[0] == kGdnValueHeads &&
               ok->scratch.o.extent[1] == kGdnHeadDim,
           "o is [48,128] FP32");
    expect(static_cast<std::byte*>(ok->scratch.u.pointer) -
                   static_cast<std::byte*>(ok->scratch.qkv.pointer) ==
               static_cast<std::ptrdiff_t>(qw38::runtime::kGdnOffU),
           "u scratch offset");
    expect(ok->scratch.u.dtype == ArithmeticDtype::Bf16 &&
               ok->scratch.u.extent[0] == kGdnValueHeads &&
               ok->scratch.u.extent[1] == kGdnHeadDim,
           "u is [48,128] BF16");
  }

  Stream empty;
  auto bad_stream = bind_gdn_front_plan(views, empty);
  expect(!bad_stream &&
             bad_stream.error().code == qw38::runtime::ErrorCode::InvalidArgument,
         "empty stream rejects");

  cursor = 3;
  auto bad_c = bind_gdn_front_plan(views, stream);
  expect(!bad_c, "cursor >= 3 rejects");
  cursor = 0;

  auto attn = views;
  attn.language_layer = 3;
  auto bad_layer = bind_gdn_front_plan(attn, stream);
  expect(!bad_layer, "attention layer rejects");

  auto idx = gdn_state_index(4);
  expect(static_cast<bool>(idx) && *idx == 3, "language 4 → GDN 3");
  expect(gdn_key_head(47) == 15, "value_head/3 boundary");

  auto shape = views;
  shape.qkv.extent[0] = 16;
  auto bad_shape = bind_gdn_front_plan(shape, stream);
  expect(!bad_shape, "qkv shape rejects");

  auto ab = views;
  ab.a.layout = PhysicalLayoutId::CudaQ4G64V0;
  auto bad_ab = bind_gdn_front_plan(ab, stream);
  expect(!bad_ab, "Q4 a/b rejects");
}

void test_conv_history_and_taps(Stream const& stream) {
  auto qkv = filled_h(kQkvWidth, 1.25f);
  auto taps = zeros_h(kConvKernel * kQkvWidth);
  taps[3 * kQkvWidth] = qw38::format::fp32_to_bf16_rne(1.0f);
  auto history = zeros_h(kConvHistoryTaps * kQkvWidth);
  auto conv = zeros_h(kQkvWidth);
  std::uint32_t cursor = 0;
  auto href = history;
  auto cref = conv;
  auto cref_cursor = cursor;
  auto rst = gdn_conv_history_step(qkv, taps, href, cref_cursor, cref);
  expect(static_cast<bool>(rst), "cpu conv");

  auto d_q = upload_vec(qkv, stream);
  auto d_t = upload_vec(taps, stream);
  auto d_h = upload_vec(history, stream);
  auto d_c = DeviceBuffer::allocate(kQkvWidth * 2);
  if (!d_q || !d_t || !d_h || !d_c) {
    fail("conv upload");
    return;
  }
  auto st = launch_gdn_conv_silu(static_cast<std::uint16_t const*>(d_q->data()),
                                 static_cast<std::uint16_t const*>(d_t->data()),
                                 static_cast<std::uint16_t*>(d_h->data()), cursor,
                                 static_cast<std::uint16_t*>(d_c->data()), stream);
  expect(static_cast<bool>(st), "conv launch");
  auto got_c = download_vec<std::uint16_t>(*d_c, kQkvWidth, stream);
  auto got_h = download_vec<std::uint16_t>(*d_h, kConvHistoryTaps * kQkvWidth, stream);
  if (!got_c || !got_h) {
    fail("conv download");
    return;
  }
  expect_bf16_close(*got_c, cref, "conv tap-current", kGdnConvBf16Abs, 0.0f);
  expect(*got_h == href, "history stores raw qkv");
  expect((*got_h)[0] == qkv[0], "SiLU output is not written to history");

  auto st_bad = launch_gdn_conv_silu(static_cast<std::uint16_t const*>(d_q->data()),
                                     static_cast<std::uint16_t const*>(d_t->data()),
                                     static_cast<std::uint16_t*>(d_h->data()), 3,
                                     static_cast<std::uint16_t*>(d_c->data()),
                                     stream);
  expect(!st_bad, "kernel rejects cursor>=3");

  auto hist0 = zeros_h(kConvHistoryTaps * kQkvWidth);
  auto d_h2 = upload_vec(hist0, stream);
  if (!d_h2) {
    fail("wrap history upload");
    return;
  }
  std::uint32_t wrap_cursor = 0;
  for (int t = 0; t < 4; ++t) {
    auto cur = filled_h(kQkvWidth, 0.5f * static_cast<float>(t + 1));
    auto d_cur = upload_vec(cur, stream);
    if (!d_cur) {
      fail("wrap upload");
      return;
    }
    st = launch_gdn_conv_silu(static_cast<std::uint16_t const*>(d_cur->data()),
                              static_cast<std::uint16_t const*>(d_t->data()),
                              static_cast<std::uint16_t*>(d_h2->data()), wrap_cursor,
                              static_cast<std::uint16_t*>(d_c->data()), stream);
    expect(static_cast<bool>(st), "wrap launch");
    wrap_cursor = (wrap_cursor + 1u) % 3u;
  }
  got_h = download_vec<std::uint16_t>(*d_h2, kConvHistoryTaps * kQkvWidth, stream);
  if (!got_h) {
    fail("wrap download");
    return;
  }
  expect(wrap_cursor == 1, "wrap cursor");
  expect(qw38::format::bf16_to_fp32((*got_h)[0]) == 2.0f, "slot 0 is token 4");
}

void test_prepare_zero_and_gates(Stream const& stream) {
  auto convolved = zeros_h(kQkvWidth);
  std::vector<float> a(kGdnValueHeads, 0.0f);
  std::vector<float> b(kGdnValueHeads, 0.0f);
  auto alog = zeros_h(kGdnValueHeads);
  auto dt = zeros_h(kGdnValueHeads);
  std::size_t const nqk = static_cast<std::size_t>(kGdnKeyHeads) * kGdnHeadDim;
  std::vector<float> q_ref(nqk, 9.0f);
  std::vector<float> k_ref(nqk, 9.0f);
  std::vector<float> alpha_ref(kGdnValueHeads, 9.0f);
  std::vector<float> beta_ref(kGdnValueHeads, 9.0f);
  auto rst = gdn_prepare(convolved, a, b, alog, dt, kDefaultRmsEps, q_ref, k_ref,
                         alpha_ref, beta_ref);
  expect(static_cast<bool>(rst), "cpu zero prepare");

  auto d_c = upload_vec(convolved, stream);
  auto d_a = upload_vec(a, stream);
  auto d_b = upload_vec(b, stream);
  auto d_al = upload_vec(alog, stream);
  auto d_dt = upload_vec(dt, stream);
  auto d_q = DeviceBuffer::allocate(nqk * 4);
  auto d_k = DeviceBuffer::allocate(nqk * 4);
  auto d_alpha = DeviceBuffer::allocate(kGdnValueHeads * 4);
  auto d_beta = DeviceBuffer::allocate(kGdnValueHeads * 4);
  if (!d_c || !d_a || !d_b || !d_al || !d_dt || !d_q || !d_k || !d_alpha ||
      !d_beta) {
    fail("prep upload");
    return;
  }
  auto st = launch_gdn_prepare(static_cast<std::uint16_t const*>(d_c->data()),
                               static_cast<float const*>(d_a->data()),
                               static_cast<float const*>(d_b->data()),
                               static_cast<std::uint16_t const*>(d_al->data()),
                               static_cast<std::uint16_t const*>(d_dt->data()),
                               kDefaultRmsEps, static_cast<float*>(d_q->data()),
                               static_cast<float*>(d_k->data()),
                               static_cast<float*>(d_alpha->data()),
                               static_cast<float*>(d_beta->data()), stream);
  expect(static_cast<bool>(st), "zero-norm launch");
  auto gq = download_vec<float>(*d_q, nqk, stream);
  auto gk = download_vec<float>(*d_k, nqk, stream);
  auto ga = download_vec<float>(*d_alpha, kGdnValueHeads, stream);
  auto gb = download_vec<float>(*d_beta, kGdnValueHeads, stream);
  if (!gq || !gk || !ga || !gb) {
    fail("zero-norm download");
    return;
  }
  expect_fp32_close(*gq, q_ref, "zero q", kGdnQkFp32Abs, kGdnQkFp32Rel);
  expect_fp32_close(*gk, k_ref, "zero k", kGdnQkFp32Abs, kGdnQkFp32Rel);
  expect_fp32_close(*ga, alpha_ref, "zero alpha", kGdnGateFp32Abs, kGdnGateFp32Rel);
  expect_fp32_close(*gb, beta_ref, "zero beta", kGdnGateFp32Abs, kGdnGateFp32Rel);

  a.assign(kGdnValueHeads, -0.5f);
  b.assign(kGdnValueHeads, 1.5f);
  alog = filled_h(kGdnValueHeads, -0.25f);
  dt = filled_h(kGdnValueHeads, 0.75f);
  rst = gdn_alpha_beta(a, b, alog, dt, alpha_ref, beta_ref);
  expect(static_cast<bool>(rst), "cpu nonzero gates");
  auto d_a2 = upload_vec(a, stream);
  auto d_b2 = upload_vec(b, stream);
  auto d_al2 = upload_vec(alog, stream);
  auto d_dt2 = upload_vec(dt, stream);
  if (!d_a2 || !d_b2 || !d_al2 || !d_dt2) {
    fail("nonzero gate upload");
    return;
  }
  st = launch_gdn_prepare(static_cast<std::uint16_t const*>(d_c->data()),
                          static_cast<float const*>(d_a2->data()),
                          static_cast<float const*>(d_b2->data()),
                          static_cast<std::uint16_t const*>(d_al2->data()),
                          static_cast<std::uint16_t const*>(d_dt2->data()),
                          kDefaultRmsEps, static_cast<float*>(d_q->data()),
                          static_cast<float*>(d_k->data()),
                          static_cast<float*>(d_alpha->data()),
                          static_cast<float*>(d_beta->data()), stream);
  expect(static_cast<bool>(st), "nonzero gate launch");
  ga = download_vec<float>(*d_alpha, kGdnValueHeads, stream);
  gb = download_vec<float>(*d_beta, kGdnValueHeads, stream);
  if (!ga || !gb) {
    fail("nonzero gate download");
    return;
  }
  expect_fp32_close(*ga, alpha_ref, "nonzero alpha", kGdnGateFp32Abs,
                    kGdnGateFp32Rel);
  expect_fp32_close(*gb, beta_ref, "nonzero beta", kGdnGateFp32Abs, kGdnGateFp32Rel);
}

void test_workspace_and_missing_model(Stream const& stream) {
  auto ws = DeviceBuffer::allocate(kGdnWorkspaceBytesPerToken);
  if (!ws) {
    fail("workspace alloc");
    return;
  }
  auto view = make_view(ws->data(), ArithmeticDtype::Fp32,
                        PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, true,
                        1, kGdnWorkspaceBytesPerToken / 4);
  auto slices = bind_gdn_workspace(view);
  expect(static_cast<bool>(slices), "workspace bind");
  if (slices) {
    expect(slices->qkv.pointer != slices->convolved.pointer, "qkv != convolved");
    expect(slices->q_hat.pointer != slices->k_hat.pointer, "q != k");
    expect(slices->alpha.pointer != slices->beta.pointer, "alpha != beta");
    expect(slices->o.pointer != slices->u.pointer, "o != u");
    expect(static_cast<std::byte*>(slices->u.pointer) -
                   static_cast<std::byte*>(slices->qkv.pointer) ==
               static_cast<std::ptrdiff_t>(qw38::runtime::kGdnOffU),
           "workspace u offset");
  }

  qw38::format::test::ScratchDir dir("qw38-gdn-unit");
  auto fx = qw38::runtime::test::write_language_fixture(dir.file("model.qw38"));
  expect(!fx.path.empty(), "write fixture");
  if (fx.path.empty()) {
    return;
  }
  auto rt = qw38::runtime::Runtime::create();
  if (!rt) {
    fail("runtime");
    return;
  }
  auto model = rt->load(fx.path);
  auto session = rt->create_session(*model, 1);
  if (!model || !session) {
    fail("load/session");
    return;
  }
  auto plan = bind_gdn_front_plan(*model, *session, 0, stream);
  expect(!plan && plan.error().code == qw38::runtime::ErrorCode::InvalidArgument,
         "missing GDN identities reject");
  auto mix = bind_gdn_plan(*model, *session, 0, stream);
  expect(!mix && mix.error().code == qw38::runtime::ErrorCode::InvalidArgument,
         "missing complete mixer identities reject");
  auto bad = bind_gdn_front_plan(*model, *session, 3, stream);
  expect(!bad, "attention layer 3 rejects");
  (void)malloc_count;
}

GdnRecurrenceBindViews dummy_recur_views(void* s, void* o, void* q, void* k,
                                         void* alpha, void* beta, void* v) {
  GdnRecurrenceBindViews views;
  std::size_t const nqk = static_cast<std::size_t>(kGdnKeyHeads) * kGdnHeadDim;
  views.q_hat = make_view(q, ArithmeticDtype::Fp32, PhysicalLayoutId::CudaFp32VectorV0,
                          StorageClass::Fp32, false, 2, kGdnKeyHeads, kGdnHeadDim);
  views.k_hat = make_view(k, ArithmeticDtype::Fp32, PhysicalLayoutId::CudaFp32VectorV0,
                          StorageClass::Fp32, false, 2, kGdnKeyHeads, kGdnHeadDim);
  views.alpha = make_view(alpha, ArithmeticDtype::Fp32,
                          PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32,
                          false, 1, kGdnValueHeads);
  views.beta = make_view(beta, ArithmeticDtype::Fp32, PhysicalLayoutId::CudaFp32VectorV0,
                         StorageClass::Fp32, false, 1, kGdnValueHeads);
  views.v = make_view(v, ArithmeticDtype::Bf16, PhysicalLayoutId::CudaBf16RowMajorV0,
                      StorageClass::Bf16, false, 2, kGdnValueHeads, kGdnHeadDim);
  views.s = gdn_state_view(s);
  views.o = make_view(o, ArithmeticDtype::Fp32, PhysicalLayoutId::CudaFp32VectorV0,
                      StorageClass::Fp32, true, 2, kGdnValueHeads, kGdnHeadDim);
  views.s_layer = 0;
  views.language_layer = 0;
  (void)nqk;
  return views;
}

void test_recurrence_bind_and_invalid(Stream const& stream) {
  std::size_t const nqk = static_cast<std::size_t>(kGdnKeyHeads) * kGdnHeadDim;
  std::size_t const nv = static_cast<std::size_t>(kGdnValueHeads) * kGdnHeadDim;
  auto d_q = DeviceBuffer::allocate(nqk * 4);
  auto d_k = DeviceBuffer::allocate(nqk * 4);
  auto d_a = DeviceBuffer::allocate(kGdnValueHeads * 4);
  auto d_b = DeviceBuffer::allocate(kGdnValueHeads * 4);
  auto d_v = DeviceBuffer::allocate(nv * 2);
  auto d_s = DeviceBuffer::allocate(kGdnSElemsPerLayer * 4);
  auto d_o = DeviceBuffer::allocate(nv * 4);
  if (!d_q || !d_k || !d_a || !d_b || !d_v || !d_s || !d_o) {
    fail("recur bind alloc");
    return;
  }
  auto views = dummy_recur_views(d_s->data(), d_o->data(), d_q->data(), d_k->data(),
                                 d_a->data(), d_b->data(), d_v->data());
  auto ok = bind_gdn_recurrence_plan(views, stream);
  expect(static_cast<bool>(ok), "valid recurrence bind");
  if (ok) {
    expect(ok->s_layer == 0 && ok->gdn_layer == 0, "layer 0 metadata");
    expect(ok->q_hat.dtype == ArithmeticDtype::Fp32, "q FP32");
    expect(ok->v.dtype == ArithmeticDtype::Bf16, "v BF16");
    expect(ok->s.dtype == ArithmeticDtype::Fp32, "S FP32");
    expect(ok->o.dtype == ArithmeticDtype::Fp32, "o FP32");
  }

  Stream empty;
  expect(!bind_gdn_recurrence_plan(views, empty), "empty stream rejects");

  auto attn = views;
  attn.language_layer = 3;
  expect(!bind_gdn_recurrence_plan(attn, stream), "attention language rejects");

  auto host = views;
  host.s.space = qw38::runtime::MemorySpace::Host;
  expect(!bind_gdn_recurrence_plan(host, stream), "host S rejects");

  auto bf16s = views;
  bf16s.s.dtype = ArithmeticDtype::Bf16;
  expect(!bind_gdn_recurrence_plan(bf16s, stream), "BF16 S rejects");

  auto transposed = views;
  transposed.s.layout = PhysicalLayoutId::CudaFp32VectorV0;
  expect(!bind_gdn_recurrence_plan(transposed, stream), "transposed S rejects");

  auto wrong_storage = views;
  wrong_storage.s.storage = StorageClass::Bf16;
  expect(!bind_gdn_recurrence_plan(wrong_storage, stream), "wrong S storage rejects");

  auto wrong_shape = views;
  wrong_shape.s.extent[2] = 127;
  expect(!bind_gdn_recurrence_plan(wrong_shape, stream), "wrong S shape rejects");

  auto overflow = views;
  overflow.s.extent[0] = std::numeric_limits<std::uint64_t>::max();
  overflow.s.extent[1] = 2;
  auto overflow_result = bind_gdn_recurrence_plan(overflow, stream);
  expect(!overflow_result &&
             overflow_result.error().code == qw38::runtime::ErrorCode::Overflow,
         "overflowing S shape rejects");

  auto same = views;
  same.o.pointer = same.s.pointer;
  expect(!bind_gdn_recurrence_plan(same, stream), "aliased S/o rejects");

  auto qk = views;
  qk.k_hat.pointer = qk.q_hat.pointer;
  expect(!bind_gdn_recurrence_plan(qk, stream), "aliased q/k rejects");

  auto wrong_layer = views;
  wrong_layer.language_layer = 4;
  expect(!bind_gdn_recurrence_plan(wrong_layer, stream), "wrong derived S layer rejects");

  auto st = launch_gdn_recurrence(nullptr, static_cast<float*>(d_k->data()),
                                  static_cast<float*>(d_a->data()),
                                  static_cast<float*>(d_b->data()),
                                  static_cast<std::uint16_t*>(d_v->data()),
                                  static_cast<float*>(d_s->data()), 0,
                                  static_cast<float*>(d_o->data()), stream);
  expect(!st, "null q_hat launch rejects");
  st = launch_gdn_recurrence(static_cast<float*>(d_q->data()),
                             static_cast<float*>(d_k->data()),
                             static_cast<float*>(d_a->data()),
                             static_cast<float*>(d_b->data()),
                             static_cast<std::uint16_t*>(d_v->data()),
                             static_cast<float*>(d_s->data()), 48,
                             static_cast<float*>(d_o->data()), stream);
  expect(!st, "s_layer 48 launch rejects");
}

void test_recurrence_layout_zero_heads_layers(Stream const& stream) {
  std::size_t const nqk = static_cast<std::size_t>(kGdnKeyHeads) * kGdnHeadDim;
  std::size_t const nv = static_cast<std::size_t>(kGdnValueHeads) * kGdnHeadDim;
  std::size_t const ns = static_cast<std::size_t>(kGdnSElemsPerLayer);

  auto q = zeros_f(static_cast<std::uint32_t>(nqk));
  auto k = zeros_f(static_cast<std::uint32_t>(nqk));
  auto alpha = filled_f(kGdnValueHeads, 0.5f);
  auto beta = filled_f(kGdnValueHeads, 1.0f);
  auto v = zeros_h(static_cast<std::uint32_t>(nv));
  auto s = zeros_f(static_cast<std::uint32_t>(ns));
  auto o = filled_f(static_cast<std::uint32_t>(nv), 9.0f);

  // One-hot key 11, value row 9 of head 5; k_hat on key head 5/3=1.
  std::uint32_t const vh = 5;
  std::uint32_t const value_j = 9;
  std::uint32_t const key = 11;
  std::uint32_t const kh = gdn_key_head(vh);
  k[kh * kGdnHeadDim + key] = 1.0f;
  q[kh * kGdnHeadDim + key] = 1.0f;
  v[vh * kGdnHeadDim + value_j] = qw38::format::fp32_to_bf16_rne(1.0f);
  for (std::uint32_t h = 0; h < kGdnValueHeads; ++h) {
    if (h != vh) {
      alpha[h] = 0.0f;
      beta[h] = 0.0f;
    }
  }

  auto s_ref = s;
  auto o_ref = o;
  auto rst = gdn_recurrence_step(q, k, alpha, beta, v, s_ref, o_ref);
  expect(static_cast<bool>(rst), "cpu one-hot");
  auto off = gdn_s_byte_offset(0, vh, value_j, key);
  expect(static_cast<bool>(off), "HVK offset");
  if (off) {
    expect(*off == gdn_s_index(vh, value_j, key) * 4u, "byte offset is [vh,value,key]");
    expect(s_ref[gdn_s_index(vh, value_j, key)] == 1.0f, "updated key is 1");
    auto trans = gdn_s_byte_offset(0, vh, key, value_j);
    if (trans && *trans != *off) {
      expect(s_ref[*trans / 4u] == 0.0f, "transposed [vh,key,value] slot stays 0");
    }
  }
  // Same warp lane owns keys 11,43,75,107; only 11 had k_hat.
  expect(s_ref[gdn_s_index(vh, value_j, 43)] == 0.0f, "lane sibling key 43 stays 0");
  expect(s_ref[gdn_s_index(7, value_j, key)] == 0.0f, "other head stays 0");

  auto d_q = upload_vec(q, stream);
  auto d_k = upload_vec(k, stream);
  auto d_a = upload_vec(alpha, stream);
  auto d_b = upload_vec(beta, stream);
  auto d_v = upload_vec(v, stream);
  auto d_s = upload_vec(s, stream);
  auto d_o = DeviceBuffer::allocate(nv * 4);
  if (!d_q || !d_k || !d_a || !d_b || !d_v || !d_s || !d_o) {
    fail("layout upload");
    return;
  }
  auto st = launch_gdn_recurrence(static_cast<float const*>(d_q->data()),
                                  static_cast<float const*>(d_k->data()),
                                  static_cast<float const*>(d_a->data()),
                                  static_cast<float const*>(d_b->data()),
                                  static_cast<std::uint16_t const*>(d_v->data()),
                                  static_cast<float*>(d_s->data()), 0,
                                  static_cast<float*>(d_o->data()), stream);
  expect(static_cast<bool>(st), "one-hot launch");
  auto got_s = download_vec<float>(*d_s, ns, stream);
  auto got_o = download_vec<float>(*d_o, nv, stream);
  if (!got_s || !got_o) {
    fail("one-hot download");
    return;
  }
  expect_fp32_close(*got_s, s_ref, "one-hot S", kGdnRecurSAbs, kGdnRecurSRel);
  expect_fp32_close(*got_o, o_ref, "one-hot o", kGdnRecurOAbs, kGdnRecurORel);

  // Zero state, nonzero q/k/v: S_new = k_hat * beta * v because p=0.
  auto qz = pattern_f(static_cast<std::uint32_t>(nqk), 0.4f);
  auto kz = pattern_f(static_cast<std::uint32_t>(nqk), -0.3f);
  auto az = filled_f(kGdnValueHeads, 0.25f);
  auto bz = filled_f(kGdnValueHeads, 0.75f);
  auto vz = pattern_h(static_cast<std::uint32_t>(nv), 0.6f);
  auto sz = zeros_f(static_cast<std::uint32_t>(ns));
  auto oz = zeros_f(static_cast<std::uint32_t>(nv));
  auto sz_ref = sz;
  auto oz_ref = oz;
  rst = gdn_recurrence_step(qz, kz, az, bz, vz, sz_ref, oz_ref);
  expect(static_cast<bool>(rst), "cpu zero-state");
  auto d_qz = upload_vec(qz, stream);
  auto d_kz = upload_vec(kz, stream);
  auto d_az = upload_vec(az, stream);
  auto d_bz = upload_vec(bz, stream);
  auto d_vz = upload_vec(vz, stream);
  auto d_sz = upload_vec(sz, stream);
  if (!d_qz || !d_kz || !d_az || !d_bz || !d_vz || !d_sz) {
    fail("zero-state upload");
    return;
  }
  st = launch_gdn_recurrence(static_cast<float const*>(d_qz->data()),
                             static_cast<float const*>(d_kz->data()),
                             static_cast<float const*>(d_az->data()),
                             static_cast<float const*>(d_bz->data()),
                             static_cast<std::uint16_t const*>(d_vz->data()),
                             static_cast<float*>(d_sz->data()), 0,
                             static_cast<float*>(d_o->data()), stream);
  expect(static_cast<bool>(st), "zero-state launch");
  got_s = download_vec<float>(*d_sz, ns, stream);
  got_o = download_vec<float>(*d_o, nv, stream);
  if (!got_s || !got_o) {
    fail("zero-state download");
    return;
  }
  expect_fp32_close(*got_s, sz_ref, "zero-state S", kGdnRecurSAbs, kGdnRecurSRel);
  expect_fp32_close(*got_o, oz_ref, "zero-state o", kGdnRecurOAbs, kGdnRecurORel);

  // Independent layers: two-layer buffer, layer 1 holds a unique fill.
  auto two = zeros_f(static_cast<std::uint32_t>(2u * ns));
  for (std::size_t i = 0; i < ns; ++i) {
    two[ns + i] = 0.125f + static_cast<float>(i % 17) * 1.0e-3f;
  }
  auto layer1 = std::vector<float>(two.begin() + static_cast<std::ptrdiff_t>(ns), two.end());
  auto d_two = upload_vec(two, stream);
  if (!d_two) {
    fail("two-layer upload");
    return;
  }
  auto views = dummy_recur_views(d_two->data(), d_o->data(), d_qz->data(), d_kz->data(),
                                 d_az->data(), d_bz->data(), d_vz->data());
  views.s_layer = 0;
  auto plan0 = bind_gdn_recurrence_plan(views, stream);
  expect(static_cast<bool>(plan0), "bind layer 0 of two");
  if (plan0) {
    expect(static_cast<bool>(execute_gdn_recurrence(*plan0)), "execute layer 0");
  }
  auto got_two = download_vec<float>(*d_two, 2u * ns, stream);
  if (!got_two) {
    fail("two-layer download");
    return;
  }
  std::vector<float> got_l1(got_two->begin() + static_cast<std::ptrdiff_t>(ns),
                            got_two->end());
  expect_fp32_close(got_l1, layer1, "layer 1 untouched", 0.0f, 0.0f);
}

void test_recurrence_reset() {
  qw38::format::test::ScratchDir dir("qw38-gdn-recur-unit");
  auto fx = qw38::runtime::test::write_language_fixture(dir.file("model.qw38"));
  expect(!fx.path.empty(), "write fixture");
  if (fx.path.empty()) {
    return;
  }
  auto rt = qw38::runtime::Runtime::create();
  if (!rt) {
    fail("runtime");
    return;
  }
  auto const& stream = rt->stream();
  auto model = rt->load(fx.path);
  auto session = rt->create_session(*model, 1);
  if (!model || !session) {
    fail("load/session");
    return;
  }
  auto plan = bind_gdn_recurrence_plan(*session, 0, stream);
  expect(static_cast<bool>(plan), "session recurrence bind");
  if (!plan) {
    return;
  }
  std::size_t const nqk = static_cast<std::size_t>(kGdnKeyHeads) * kGdnHeadDim;
  std::size_t const nv = static_cast<std::size_t>(kGdnValueHeads) * kGdnHeadDim;
  auto q = pattern_f(static_cast<std::uint32_t>(nqk), 0.2f);
  auto k = pattern_f(static_cast<std::uint32_t>(nqk), 0.3f);
  auto a = filled_f(kGdnValueHeads, 0.8f);
  auto b = filled_f(kGdnValueHeads, 0.4f);
  auto v = pattern_h(static_cast<std::uint32_t>(nv), 0.5f);
  expect(static_cast<bool>(qw38::cuda::copy_h2d(plan->q_hat.pointer, q.data(),
                                               nqk * 4, stream)),
         "upload q");
  expect(static_cast<bool>(qw38::cuda::copy_h2d(plan->k_hat.pointer, k.data(),
                                               nqk * 4, stream)),
         "upload k");
  expect(static_cast<bool>(qw38::cuda::copy_h2d(plan->alpha.pointer, a.data(),
                                               kGdnValueHeads * 4, stream)),
         "upload alpha");
  expect(static_cast<bool>(qw38::cuda::copy_h2d(plan->beta.pointer, b.data(),
                                               kGdnValueHeads * 4, stream)),
         "upload beta");
  expect(static_cast<bool>(qw38::cuda::copy_h2d(plan->v.pointer, v.data(), nv * 2,
                                               stream)),
         "upload v");
  auto const mallocs = malloc_count();
  expect(static_cast<bool>(execute_gdn_recurrence(*plan)), "first execute");
  expect(malloc_count() == mallocs, "recurrence allocates nothing");
  expect(static_cast<bool>(session->reset()), "reset");
  std::vector<float> s0(8, 1.0f);
  auto off = gdn_s_byte_offset(0, 0, 0, 0);
  expect(static_cast<bool>(off), "s offset");
  if (off) {
    expect(static_cast<bool>(qw38::cuda::copy_d2h(
               s0.data(), static_cast<std::byte*>(session->gdn_s().pointer) + *off,
               s0.size() * 4, stream)),
           "download S after reset");
    expect(static_cast<bool>(stream.sync()), "sync reset");
    bool z = true;
    for (float x : s0) {
      z = z && (x == 0.0f);
    }
    expect(z, "reset zeros S");
  }
  auto s_ref = zeros_f(static_cast<std::uint32_t>(kGdnSElemsPerLayer));
  auto o_ref = zeros_f(static_cast<std::uint32_t>(nv));
  auto rst = gdn_recurrence_step(q, k, a, b, v, s_ref, o_ref);
  expect(static_cast<bool>(rst), "cpu after reset");
  expect(static_cast<bool>(execute_gdn_recurrence(*plan)), "execute after reset");
  std::vector<float> o_got(nv);
  expect(static_cast<bool>(qw38::cuda::copy_d2h(o_got.data(), plan->o.pointer,
                                               nv * 4, stream)),
         "download o");
  expect(static_cast<bool>(stream.sync()), "sync o");
  expect_fp32_close(o_got, o_ref, "o after reset", kGdnRecurOAbs, kGdnRecurORel);
}

GdnBindViews dummy_mixer_views(std::uint32_t* cursor) {
  GdnBindViews v;
  auto front = dummy_ok_views(cursor);
  v.qkv = front.qkv;
  v.qkv_scales = front.qkv_scales;
  v.z = front.z;
  v.z_scales = front.z_scales;
  v.a = front.a;
  v.b = front.b;
  v.gamma = front.gamma;
  v.taps = front.taps;
  v.a_log = front.a_log;
  v.dt_bias = front.dt_bias;
  v.residual = front.residual;
  v.normalized = front.normalized;
  v.workspace = front.workspace;
  v.history = front.history;
  v.cursor = front.cursor;
  v.language_layer = front.language_layer;
  v.out = make_view(dummy_ptr(0x1E0000), ArithmeticDtype::Bf16,
                    PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped, false, 2,
                    kHidden, kGdnZWidth);
  std::uint64_t const out_scales =
      static_cast<std::uint64_t>(kHidden) * (kGdnZWidth / 64u);
  v.out_scales = make_view(dummy_ptr(0x1F0000), ArithmeticDtype::Fp16,
                           PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped,
                           false, 1, out_scales);
  v.gated_gamma = make_view(dummy_ptr(0x200000), ArithmeticDtype::Bf16,
                            PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16,
                            false, 1, kGdnHeadDim);
  v.residual_out = make_view(dummy_ptr(0x210000), ArithmeticDtype::Fp32,
                             PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32,
                             true, 1, kHidden);
  v.s = gdn_state_view(dummy_ptr(0x220000));
  return v;
}

bool make_q4_mixer(HostGdnMixer& host, std::int8_t qkv_code, std::int8_t z_code,
                   std::int8_t out_code, float h_seed, bool zero_gated,
                   bool zero_z) {
  auto lq = make_logical(LogicalQuantizerId::Q4G64V0, kQkvWidth, kHidden, qkv_code,
                         0x3C00);
  auto lz = make_logical(LogicalQuantizerId::Q4G64V0, kGdnZWidth, kHidden,
                         zero_z ? std::int8_t{0} : z_code, 0x3C00);
  auto lo = make_logical(LogicalQuantizerId::Q4G64V0, kHidden, kGdnZWidth, out_code,
                         0x3C00);
  fill_logical_pattern(lq, 2);
  if (!zero_z) {
    fill_logical_pattern(lz, 5);
  }
  fill_logical_pattern(lo, 9);
  if (!pack_q4_from_logical(lq, host.qkv, "qkv") ||
      !pack_q4_from_logical(lz, host.z, "z") ||
      !pack_q4_from_logical(lo, host.out, "out")) {
    return false;
  }
  auto wq = qw38::compiler::dequantize_to_bf16(lq);
  auto wz = qw38::compiler::dequantize_to_bf16(lz);
  auto wo = qw38::compiler::dequantize_to_bf16(lo);
  if (!wq || !wz || !wo) {
    fail("mixer dequant");
    return false;
  }
  host.w_qkv = std::move(*wq);
  host.w_z = std::move(*wz);
  host.w_out = std::move(*wo);
  host.w_a = pattern_h(kGdnValueHeads * kHidden, 0.04f);
  host.w_b = pattern_h(kGdnValueHeads * kHidden, -0.03f);
  if (!pack_bf16_tile(host.w_a, kGdnValueHeads, kHidden, host.a, "a") ||
      !pack_bf16_tile(host.w_b, kGdnValueHeads, kHidden, host.b, "b")) {
    return false;
  }
  host.gamma = pattern_h(kHidden, 0.2f);
  host.gated_gamma = zero_gated ? zeros_h(kGdnHeadDim) : pattern_h(kGdnHeadDim, 0.7f);
  host.taps = pattern_h(kConvKernel * kQkvWidth, 0.15f);
  host.a_log = pattern_h(kGdnValueHeads, -0.5f);
  host.dt_bias = pattern_h(kGdnValueHeads, 0.4f);
  host.residual = residual_vec(kHidden, h_seed);
  return true;
}

void test_mixer_bind_and_lifetimes(Stream const& stream) {
  std::uint32_t cursor = 0;
  auto views = dummy_mixer_views(&cursor);
  auto ok = bind_gdn_plan(views, stream);
  expect(static_cast<bool>(ok), "valid mixer dummy bind");
  if (ok) {
    expect(ok->front.scratch.u.extent[0] == kGdnValueHeads, "plan u heads");
    expect(ok->residual_out.pointer != ok->front.residual.pointer,
           "h and h_mid distinct");
    expect(ok->s_layer == 0, "layer 0 s_layer");
  }

  Stream empty;
  expect(!bind_gdn_plan(views, empty), "empty stream rejects mixer");

  auto alias = views;
  alias.residual_out.pointer = alias.residual.pointer;
  expect(!bind_gdn_plan(alias, stream), "aliased residual/h_mid rejects");

  auto same_g = views;
  same_g.gated_gamma.pointer = same_g.gamma.pointer;
  expect(!bind_gdn_plan(same_g, stream), "aliased gammas reject");

  auto attn = views;
  attn.language_layer = 3;
  expect(!bind_gdn_plan(attn, stream), "attention layer rejects mixer");

  auto tiny_s = views;
  tiny_s.s.extent[0] = 16;
  expect(!bind_gdn_plan(tiny_s, stream), "short S rejects mixer");

  auto mix_fam = views;
  mix_fam.out.layout = PhysicalLayoutId::CudaBf16DenseTileV0;
  mix_fam.out.storage = StorageClass::Bf16;
  mix_fam.out_scales = {};
  expect(!bind_gdn_plan(mix_fam, stream), "mixed Q4/BF16 out family rejects");
}

void test_mixer_gamma_z_residual_state(Stream const& stream) {
  HostGdnMixer host;
  if (!make_q4_mixer(host, 2, -1, 3, 0.55f, true, false)) {
    return;
  }
  DeviceGdnMixer dev;
  if (!upload_host_mixer(host, dev, stream, 2)) {
    return;
  }
  auto layer1 = pattern_f(static_cast<std::uint32_t>(kGdnSElemsPerLayer), 0.11f);
  expect(static_cast<bool>(qw38::cuda::copy_h2d(
             static_cast<std::byte*>(dev.s.data()) + kGdnSElemsPerLayer * 4u,
             layer1.data(), layer1.size() * 4u, stream)),
         "seed layer 1 S");

  auto views = mixer_views(dev);
  auto plan = bind_gdn_plan(views, stream);
  expect(static_cast<bool>(plan), "gamma0 mixer bind");
  if (!plan) {
    return;
  }
  auto const mallocs = malloc_count();
  auto out = execute_decode_gdn(*plan);
  expect(static_cast<bool>(out), "gamma0 execute");
  expect(malloc_count() == mallocs, "mixer allocates nothing");
  if (!out) {
    return;
  }
  expect(out->pointer == dev.residual_out.data(), "returns h_mid");

  auto hin = download_vec<float>(dev.residual, kHidden, stream);
  auto hout = download_vec<float>(dev.residual_out, kHidden, stream);
  if (!hin || !hout) {
    fail("gamma0 download");
    return;
  }
  expect_fp32_close(*hin, host.residual, "input residual preserved", 0.0f, 0.0f);
  expect_fp32_close(*hout, host.residual, "gamma=0 Mix is 0",
                    qw38::reference::tol::kGdnMixerResidualAbs,
                    qw38::reference::tol::kGdnMixerResidualRel);

  auto s_all = download_vec<float>(dev.s, 2u * kGdnSElemsPerLayer, stream);
  if (!s_all) {
    fail("S download");
    return;
  }
  std::vector<float> got_l1(s_all->begin() + static_cast<std::ptrdiff_t>(kGdnSElemsPerLayer),
                            s_all->end());
  expect_fp32_close(got_l1, layer1, "other S layer isolated", 0.0f, 0.0f);

  HostGdnMixer zhost;
  if (!make_q4_mixer(zhost, 2, 0, 3, 0.62f, false, true)) {
    return;
  }
  DeviceGdnMixer zdev;
  if (!upload_host_mixer(zhost, zdev, stream, 1)) {
    return;
  }
  auto zviews = mixer_views(zdev);
  auto zplan = bind_gdn_plan(zviews, stream);
  expect(static_cast<bool>(zplan), "z=0 mixer bind");
  if (!zplan) {
    return;
  }
  auto zout = execute_decode_gdn(*zplan);
  expect(static_cast<bool>(zout), "z=0 execute");
  auto zhin = download_vec<float>(zdev.residual, kHidden, stream);
  auto zhout = download_vec<float>(zdev.residual_out, kHidden, stream);
  if (!zhin || !zhout) {
    fail("z=0 download");
    return;
  }
  expect_fp32_close(*zhin, zhost.residual, "z=0 input residual preserved", 0.0f, 0.0f);
  expect_fp32_close(*zhout, zhost.residual, "SiLU(z)=0 Mix is 0",
                    qw38::reference::tol::kGdnMixerResidualAbs,
                    qw38::reference::tol::kGdnMixerResidualRel);

  auto const mallocs2 = malloc_count();
  expect(static_cast<bool>(execute_decode_gdn(*zplan)), "repeated mixer");
  expect(malloc_count() == mallocs2, "repeated mixer allocates nothing");
}

}  // namespace

int main() {
  auto stream = Stream::create();
  if (!stream) {
    fail("stream");
    return 1;
  }
  test_bind_errors(*stream);
  test_conv_history_and_taps(*stream);
  test_prepare_zero_and_gates(*stream);
  test_workspace_and_missing_model(*stream);
  test_recurrence_bind_and_invalid(*stream);
  test_recurrence_layout_zero_heads_layers(*stream);
  test_recurrence_reset();
  test_mixer_bind_and_lifetimes(*stream);
  test_mixer_gamma_z_residual_state(*stream);
  if (g_failures != 0) {
    std::cerr << g_failures << " gdn unit failures\n";
    return 1;
  }
  std::cout << "gdn unit ok\n";
  return 0;
}
