#include "gdn_support.hpp"

#include "cuda/alloc.hpp"
#include "runtime/runtime.hpp"
#include "runtime_support.hpp"

#include <cstdint>
#include <iostream>
#include <string>

using qw38::cuda::DeviceBuffer;
using qw38::cuda::Stream;
using qw38::cuda::launch_gdn_conv_silu;
using qw38::cuda::launch_gdn_prepare;
using qw38::cuda::malloc_count;
using qw38::format::ArithmeticDtype;
using qw38::format::PhysicalLayoutId;
using qw38::format::StorageClass;
using qw38::gdn::test::expect;
using qw38::gdn::test::expect_bf16_close;
using qw38::gdn::test::expect_fp32_close;
using qw38::gdn::test::fail;
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
using qw38::gdn::test::kGdnValueHeads;
using qw38::gdn::test::kGdnZWidth;
using qw38::gdn::test::kHidden;
using qw38::gdn::test::kQkvWidth;
using qw38::gdn::test::make_view;
using qw38::gdn::test::pattern_h;
using qw38::gdn::test::zeros_h;
using qw38::gdn::test::download_vec;
using qw38::gdn::test::upload_vec;
using qw38::reference::gdn_alpha_beta;
using qw38::reference::gdn_conv_history_step;
using qw38::reference::gdn_key_head;
using qw38::reference::gdn_prepare;
using qw38::runtime::GdnFrontBindViews;
using qw38::runtime::bind_gdn_front_plan;
using qw38::runtime::bind_gdn_workspace;
using qw38::runtime::execute_gdn_front;
using qw38::runtime::gdn_state_index;
using qw38::runtime::kGdnBytesQHat;
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
  v.host_cursor = cursor;
  v.language_layer = 0;
  return v;
}

void test_bind_errors(Stream const& stream) {
  std::uint32_t cursor = 0;
  auto views = dummy_ok_views(&cursor);
  auto ok = bind_gdn_front_plan(views, stream);
  expect(static_cast<bool>(ok), "valid dummy bind");
  if (ok) {
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
  auto bad = bind_gdn_front_plan(*model, *session, 3, stream);
  expect(!bad, "attention layer 3 rejects");
  (void)malloc_count;
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
  if (g_failures != 0) {
    std::cerr << g_failures << " gdn unit failures\n";
    return 1;
  }
  std::cout << "gdn unit ok\n";
  return 0;
}
