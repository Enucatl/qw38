#include "mlp_support.hpp"
#include "decode_mmv_support.hpp"

#include "cuda/error.hpp"
#include "runtime/runtime.hpp"
#include "runtime_support.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>

using qw38::cuda::DecodeEpilogue;
using qw38::cuda::DecodeMmvDesc;
using qw38::cuda::DecodeMmvPairedDesc;
using qw38::cuda::DeviceBuffer;
using qw38::cuda::Stream;
using qw38::cuda::kDecodeLayoutQ4G64V0;
using qw38::cuda::kDecodeQuantizerQ4G64V0;
using qw38::cuda::launch_decode_mmv;
using qw38::cuda::launch_decode_mmv_paired;
using qw38::mlp::test::DeviceMlp;
using qw38::mlp::test::HostMlp;
using qw38::mlp::test::all_finite;
using qw38::mlp::test::bind_views;
using qw38::mlp::test::bf16_vec;
using qw38::mlp::test::download_vec;
using qw38::mlp::test::expect;
using qw38::mlp::test::expect_bf16_close;
using qw38::mlp::test::fail;
using qw38::mlp::test::fill_logical_pattern;
using qw38::mlp::test::g_failures;
using qw38::mlp::test::make_logical;
using qw38::mlp::test::make_view;
using qw38::mlp::test::pack_q4_from_logical;
using qw38::mlp::test::residual_vec;
using qw38::mlp::test::upload_host_mlp;
using qw38::mlp::test::upload_vec;
using qw38::format::LogicalQuantizerId;
using qw38::format::PhysicalLayoutId;
using qw38::format::StorageClass;
using qw38::format::fp32_to_bf16_rne;
using qw38::reference::kDefaultRmsEps;
using qw38::reference::kFfn;
using qw38::reference::kHidden;
using qw38::reference::silu_fp32;
using qw38::runtime::MlpBindViews;
using qw38::runtime::bind_mlp_plan;
using qw38::runtime::execute_decode_mlp;

namespace {

void* dummy_ptr(std::uintptr_t v) {
  return reinterpret_cast<void*>(v);
}

MlpBindViews dummy_ok_views() {
  MlpBindViews v;
  auto* p = dummy_ptr(0x10000000);
  auto* p2 = dummy_ptr(0x20000000);
  auto* p3 = dummy_ptr(0x30000000);
  auto* p4 = dummy_ptr(0x40000000);
  auto* p5 = dummy_ptr(0x50000000);
  auto* p6 = dummy_ptr(0x60000000);
  auto* p7 = dummy_ptr(0x70000000);
  auto* p8 = dummy_ptr(0x80000000);
  auto* p9 = dummy_ptr(0x90000000);
  auto* p10 = dummy_ptr(0xA0000000);
  auto* p11 = dummy_ptr(0xB0000000);
  v.gate = make_view(p, qw38::format::ArithmeticDtype::Bf16,
                     PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped,
                     false, 2, kFfn, kHidden);
  v.up = make_view(p2, qw38::format::ArithmeticDtype::Bf16,
                   PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped, false,
                   2, kFfn, kHidden);
  v.down = make_view(p3, qw38::format::ArithmeticDtype::Bf16,
                     PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped,
                     false, 2, kHidden, kFfn);
  std::uint64_t const g_scales = static_cast<std::uint64_t>(kFfn) * (kHidden / 64u);
  std::uint64_t const d_scales = static_cast<std::uint64_t>(kHidden) * (kFfn / 64u);
  v.gate_scales = make_view(p4, qw38::format::ArithmeticDtype::Fp16,
                            PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped,
                            false, 1, g_scales);
  v.up_scales = make_view(p5, qw38::format::ArithmeticDtype::Fp16,
                          PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped,
                          false, 1, g_scales);
  v.down_scales = make_view(p6, qw38::format::ArithmeticDtype::Fp16,
                            PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped,
                            false, 1, d_scales);
  v.gamma = make_view(p7, qw38::format::ArithmeticDtype::Bf16,
                      PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16, false,
                      1, kHidden);
  v.h_mid = make_view(p8, qw38::format::ArithmeticDtype::Fp32,
                      PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32,
                      true, 1, kHidden);
  v.next_h = make_view(p11, qw38::format::ArithmeticDtype::Fp32,
                       PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32,
                       true, 1, kHidden);
  v.normalized = make_view(p9, qw38::format::ArithmeticDtype::Bf16,
                           PhysicalLayoutId::CudaBf16RowMajorV0, StorageClass::Bf16,
                           true, 1, kHidden);
  v.swiglu = make_view(p10, qw38::format::ArithmeticDtype::Bf16,
                       PhysicalLayoutId::CudaBf16RowMajorV0, StorageClass::Bf16,
                       true, 1, kFfn);
  return v;
}

void test_bind_alias_errors(Stream const& stream) {
  struct ViewField {
    char const* name;
    void const* (*get)(MlpBindViews const&);
    void (*set)(MlpBindViews&, void*);
  };
#define VIEW_FIELD(member)                                                   \
  ViewField {                                                               \
    #member,                                                               \
        [](MlpBindViews const& v) -> void const* { return v.member.pointer; }, \
        [](MlpBindViews& v, void* pointer) { v.member.pointer = pointer; }   \
  }
  constexpr std::array fields{
      VIEW_FIELD(gate),       VIEW_FIELD(gate_scales), VIEW_FIELD(up),
      VIEW_FIELD(up_scales),  VIEW_FIELD(down),        VIEW_FIELD(down_scales),
      VIEW_FIELD(gamma),      VIEW_FIELD(h_mid),       VIEW_FIELD(next_h),
      VIEW_FIELD(normalized), VIEW_FIELD(swiglu),
  };
#undef VIEW_FIELD

  for (std::size_t i = 0; i < fields.size(); ++i) {
    for (std::size_t j = i + 1; j < fields.size(); ++j) {
      for (std::uintptr_t offset : {std::uintptr_t{0}, std::uintptr_t{16}}) {
        auto views = dummy_ok_views();
        auto const base =
            reinterpret_cast<std::uintptr_t>(fields[i].get(views));
        fields[j].set(views, dummy_ptr(base + offset));
        auto got = bind_mlp_plan(views, stream);
        expect(!got &&
                   got.error().code == qw38::runtime::ErrorCode::InvalidArgument,
               std::string(fields[i].name) + "/" + fields[j].name +
                   (offset == 0 ? " exact alias rejects"
                                : " offset overlap rejects"));
      }
    }
  }

  auto overflow = dummy_ok_views();
  overflow.swiglu.pointer = dummy_ptr(
      std::numeric_limits<std::uintptr_t>::max() - std::uintptr_t{15});
  auto bad_overflow = bind_mlp_plan(overflow, stream);
  expect(!bad_overflow &&
             bad_overflow.error().code == qw38::runtime::ErrorCode::InvalidArgument,
         "overflowing byte interval rejects");

  auto adjacent = dummy_ok_views();
  adjacent.next_h.pointer = dummy_ptr(
      reinterpret_cast<std::uintptr_t>(adjacent.gamma.pointer) + kHidden * 2u);
  expect(static_cast<bool>(bind_mlp_plan(adjacent, stream)),
         "adjacent byte intervals do not overlap");
}

void test_bind_errors(Stream const& stream) {
  auto views = dummy_ok_views();
  auto ok = bind_mlp_plan(views, stream);
  expect(static_cast<bool>(ok), "valid dummy bind");

  Stream empty;
  auto bad_stream = bind_mlp_plan(views, empty);
  expect(!bad_stream && bad_stream.error().code == qw38::runtime::ErrorCode::InvalidArgument,
         "empty stream rejects");

  auto bad_eps = bind_mlp_plan(views, stream, 0.0f);
  expect(!bad_eps && bad_eps.error().code == qw38::runtime::ErrorCode::InvalidArgument,
         "non-positive eps rejects");

  auto shape = views;
  shape.gate.extent[0] = 16;
  auto bad_shape = bind_mlp_plan(shape, stream);
  expect(!bad_shape && bad_shape.error().code == qw38::runtime::ErrorCode::InvalidArgument,
         "gate shape rejects");

  auto payload_dtype = views;
  payload_dtype.gate.dtype = qw38::format::ArithmeticDtype::Fp32;
  expect(!bind_mlp_plan(payload_dtype, stream),
         "weight payload arithmetic dtype rejects");

  auto layout = views;
  layout.gate.layout = PhysicalLayoutId::CudaQ8G32V0;
  layout.gate.storage = StorageClass::Int8Grouped;
  auto bad_layout = bind_mlp_plan(layout, stream);
  expect(!bad_layout && bad_layout.error().code == qw38::runtime::ErrorCode::InvalidArgument,
         "Q8 layout rejects");

  auto mix = views;
  mix.down.layout = PhysicalLayoutId::CudaBf16DenseTileV0;
  mix.down.storage = StorageClass::Bf16;
  mix.down_scales = {};
  auto bad_mix = bind_mlp_plan(mix, stream);
  expect(!bad_mix && bad_mix.error().code == qw38::runtime::ErrorCode::InvalidArgument,
         "mixed Q4/BF16 family rejects");

  auto scale_dtype = views;
  scale_dtype.gate_scales.dtype = qw38::format::ArithmeticDtype::Bf16;
  expect(!bind_mlp_plan(scale_dtype, stream), "Q4 scales require FP16 dtype");

  auto scale_layout = views;
  scale_layout.up_scales.layout = PhysicalLayoutId::CudaBf16VectorV0;
  expect(!bind_mlp_plan(scale_layout, stream),
         "Q4 scales require matching physical layout");

  auto scale_storage = views;
  scale_storage.down_scales.storage = StorageClass::Bf16;
  expect(!bind_mlp_plan(scale_storage, stream),
         "Q4 scales require grouped storage");

  auto scale_rank = views;
  scale_rank.gate_scales.rank = 2;
  scale_rank.gate_scales.extent = {1, scale_rank.gate_scales.extent[0]};
  expect(!bind_mlp_plan(scale_rank, stream), "flattened Q4 scale rank rejects");

  auto scale_extent = views;
  --scale_extent.gate_scales.extent[0];
  expect(!bind_mlp_plan(scale_extent, stream), "undersized Q4 scale rejects");

  auto vector_layout = views;
  vector_layout.gamma.layout = PhysicalLayoutId::CudaBf16RowMajorV0;
  expect(!bind_mlp_plan(vector_layout, stream), "wrong gamma vector layout rejects");

  auto vector_storage = views;
  vector_storage.next_h.storage = StorageClass::Bf16;
  expect(!bind_mlp_plan(vector_storage, stream), "wrong output storage rejects");

  auto vector_rank = views;
  vector_rank.normalized.rank = 2;
  vector_rank.normalized.extent = {1, kHidden};
  expect(!bind_mlp_plan(vector_rank, stream),
         "flattened-equivalent scratch rank rejects");

  auto oversized = views;
  ++oversized.h_mid.extent[0];
  expect(!bind_mlp_plan(oversized, stream), "oversized residual view rejects");

  auto small = views;
  small.swiglu.extent[0] = 8;
  auto bad_sw = bind_mlp_plan(small, stream);
  expect(!bad_sw && bad_sw.error().code == qw38::runtime::ErrorCode::InvalidArgument,
         "short SwiGLU view rejects");

  auto byte_overflow = views;
  byte_overflow.swiglu.extent[0] = std::numeric_limits<std::uint64_t>::max();
  auto bad_overflow = bind_mlp_plan(byte_overflow, stream);
  expect(!bad_overflow &&
             bad_overflow.error().code == qw38::runtime::ErrorCode::Overflow,
         "overflowing vector byte extent rejects");
}

void test_swiglu_epilogue(Stream const& stream) {
  DecodeMmvDesc d;
  d.layout = kDecodeLayoutQ4G64V0;
  d.quantizer = kDecodeQuantizerQ4G64V0;
  d.n = 8;
  d.k = 256;
  d.padded_n = 8;
  d.padded_k = 256;
  auto buf = DeviceBuffer::allocate(4096);
  expect(static_cast<bool>(buf), "tiny buffer");
  if (!buf) {
    return;
  }
  d.codes = qw38::cuda::decode_matrix_view(
      buf->data(), qw38::cuda::DecodeDtype::Q4, kDecodeLayoutQ4G64V0,
      8, 256, 8, 256,
      qw38::cuda::decode_code_bytes(kDecodeLayoutQ4G64V0, 8, 256), 16);
  d.scales = qw38::cuda::decode_matrix_view(
      static_cast<std::byte*>(buf->data()) + 2048,
      qw38::cuda::DecodeDtype::Fp16, kDecodeLayoutQ4G64V0,
      8, 4, 8, 4,
      qw38::cuda::decode_scale_bytes(kDecodeLayoutQ4G64V0, 8, 256), 2);
  qw38::decode_mmv::test::bind_input(
      d, static_cast<std::byte*>(buf->data()) + 2112);
  qw38::decode_mmv::test::bind_output(
      d, static_cast<std::byte*>(buf->data()) + 3072);
  d.epilogue = DecodeEpilogue::SwigluStoreBf16;
  auto unpaired = launch_decode_mmv(d, stream);
  expect(!unpaired && unpaired.error().code == qw38::cuda::ErrorCode::InvalidArgument,
         "unpaired SwiGLU epilogue rejects");

  auto logical_g = make_logical(LogicalQuantizerId::Q4G64V0, 8, 256, 3, 0x3C00);
  auto logical_u = make_logical(LogicalQuantizerId::Q4G64V0, 8, 256, -2, 0x3C00);
  auto pg = qw38::format::pack_cuda_v0(LogicalQuantizerId::Q4G64V0,
                                       PhysicalLayoutId::CudaQ4G64V0, logical_g);
  auto pu = qw38::format::pack_cuda_v0(LogicalQuantizerId::Q4G64V0,
                                       PhysicalLayoutId::CudaQ4G64V0, logical_u);
  auto x = bf16_vec(256, 0.25f);
  if (!pg || !pu) {
    fail("small swiglu pack");
    return;
  }
  auto rg = qw38::compiler::reference_gemv_packed(*pg, x);
  auto ru = qw38::compiler::reference_gemv_packed(*pu, x);
  auto d_cg = upload_vec(pg->codes, stream);
  auto d_sg = upload_vec(pg->scales, stream);
  auto d_cu = upload_vec(pu->codes, stream);
  auto d_su = upload_vec(pu->scales, stream);
  auto d_x = upload_vec(x, stream);
  auto d_y = DeviceBuffer::allocate(8 * 2);
  if (!rg || !ru || !d_cg || !d_sg || !d_cu || !d_su || !d_x || !d_y) {
    fail("small swiglu setup");
    return;
  }
  DecodeMmvPairedDesc paired;
  paired.a = qw38::decode_mmv::test::desc_from_packed(
      *pg, DecodeEpilogue::SwigluStoreBf16);
  paired.a.codes.pointer = d_cg->as_bytes();
  paired.a.scales.pointer = d_sg->as_bytes();
  qw38::decode_mmv::test::bind_input(paired.a, d_x->data());
  qw38::decode_mmv::test::bind_output(paired.a, d_y->data());
  paired.b = qw38::decode_mmv::test::desc_from_packed(
      *pu, DecodeEpilogue::SwigluStoreBf16);
  paired.b.codes.pointer = d_cu->as_bytes();
  paired.b.scales.pointer = d_su->as_bytes();
  paired.b.input = paired.a.input;
  auto st = launch_decode_mmv_paired(paired, stream);
  expect(static_cast<bool>(st), "small swiglu launch");
  auto got = download_vec<std::uint16_t>(*d_y, 8, stream);
  if (!got) {
    fail("small swiglu download");
    return;
  }
  std::vector<std::uint16_t> want(8);
  for (std::uint32_t i = 0; i < 8; ++i) {
    want[i] = fp32_to_bf16_rne(silu_fp32((*rg)[i]) * (*ru)[i]);
  }
  expect_bf16_close(*got, want, "small swiglu", 8.0e-3f, 1.0e-4f);
}

bool make_q4_host(HostMlp& host, std::int8_t gate_code, std::int8_t up_code,
                  std::int8_t down_code, float h_seed, float g_seed,
                  bool patterned) {
  auto lg = make_logical(LogicalQuantizerId::Q4G64V0, kFfn, kHidden, gate_code,
                         0x3C00);
  auto lu = make_logical(LogicalQuantizerId::Q4G64V0, kFfn, kHidden, up_code,
                         0x3C00);
  auto ld = make_logical(LogicalQuantizerId::Q4G64V0, kHidden, kFfn, down_code,
                         0x3C00);
  if (patterned) {
    fill_logical_pattern(lg, 1);
    fill_logical_pattern(lu, 4);
    fill_logical_pattern(ld, 9);
  }
  if (!pack_q4_from_logical(lg, host.gate, "gate") ||
      !pack_q4_from_logical(lu, host.up, "up") ||
      !pack_q4_from_logical(ld, host.down, "down")) {
    return false;
  }
  auto wg = qw38::compiler::dequantize_to_bf16(lg);
  auto wu = qw38::compiler::dequantize_to_bf16(lu);
  auto wd = qw38::compiler::dequantize_to_bf16(ld);
  if (!wg || !wu || !wd) {
    fail("dequant");
    return false;
  }
  host.w_gate = std::move(*wg);
  host.w_up = std::move(*wu);
  host.w_down = std::move(*wd);
  host.gamma = bf16_vec(kHidden, g_seed);
  host.h_mid = residual_vec(kHidden, h_seed);
  return true;
}

void test_zero_extreme_and_reuse(Stream const& stream) {
  HostMlp host;
  if (!make_q4_host(host, 0, 0, 0, 0.0f, 0.1f, false)) {
    return;
  }
  DeviceMlp dev;
  if (!upload_host_mlp(host, dev, stream)) {
    return;
  }
  auto views = bind_views(dev);
  auto plan = bind_mlp_plan(views, stream);
  expect(static_cast<bool>(plan), "zero bind");
  if (!plan) {
    return;
  }
  auto const mallocs = qw38::cuda::malloc_count();
  auto st = execute_decode_mlp(*plan);
  expect(static_cast<bool>(st), "zero execute");
  st = execute_decode_mlp(*plan);
  expect(static_cast<bool>(st), "zero execute 2");
  expect(qw38::cuda::malloc_count() == mallocs, "repeated call allocates nothing");
  auto got = download_vec<float>(dev.next_h, kHidden, stream);
  if (got) {
    expect(all_finite(*got, "zero residual"), "zero residual finite");
    for (auto v : *got) {
      if (v != 0.0f) {
        fail("zero residual expected 0");
        break;
      }
    }
  } else {
    fail("zero residual download");
  }

  HostMlp extreme;
  if (!make_q4_host(extreme, 7, -7, 3, 50.0f, 1.0f, true)) {
    return;
  }
  DeviceMlp dev_x;
  if (!upload_host_mlp(extreme, dev_x, stream)) {
    return;
  }
  auto views_x = bind_views(dev_x);
  auto plan_x = bind_mlp_plan(views_x, stream);
  expect(static_cast<bool>(plan_x), "extreme bind");
  if (!plan_x) {
    return;
  }
  auto const mallocs_x = qw38::cuda::malloc_count();
  st = execute_decode_mlp(*plan_x);
  expect(static_cast<bool>(st), "extreme execute");
  expect(qw38::cuda::malloc_count() == mallocs_x, "extreme execute allocates nothing");
  auto got_x = download_vec<float>(dev_x.next_h, kHidden, stream);
  if (got_x) {
    expect(all_finite(*got_x, "extreme residual"), "extreme residual finite");
  } else {
    fail("extreme residual download");
  }

  expect(plan_x->normalized.pointer == dev_x.normalized.data(),
         "normalized scratch identity");
  expect(plan_x->swiglu.pointer == dev_x.swiglu.data(), "swiglu scratch identity");
  auto const nptr = plan_x->normalized.pointer;
  auto const sptr = plan_x->swiglu.pointer;
  st = execute_decode_mlp(*plan_x);
  expect(plan_x->normalized.pointer == nptr && plan_x->swiglu.pointer == sptr,
         "scratch pointers stable across calls");
}

void test_model_bind_missing(Stream const& stream) {
  qw38::format::test::ScratchDir dir("qw38-mlp-unit");
  auto fx = qw38::runtime::test::write_language_fixture(dir.file("model.qw38"));
  expect(!fx.path.empty(), "write fixture");
  if (fx.path.empty()) {
    return;
  }
  auto rt = qw38::runtime::Runtime::create();
  expect(static_cast<bool>(rt), "runtime");
  if (!rt) {
    return;
  }
  auto model = rt->load(fx.path);
  expect(static_cast<bool>(model), "load fixture");
  if (!model) {
    return;
  }
  auto session = rt->create_session(*model, 1);
  expect(static_cast<bool>(session), "session");
  if (!session) {
    return;
  }
  auto plan = bind_mlp_plan(*model, *session, 0, stream);
  expect(!plan && plan.error().code == qw38::runtime::ErrorCode::InvalidArgument,
         "missing MLP tensor identities reject");
  auto bad_layer = bind_mlp_plan(*model, *session, 64, stream);
  expect(!bad_layer && bad_layer.error().code == qw38::runtime::ErrorCode::InvalidArgument,
         "layer>=64 rejects");
}

}  // namespace

int main() {
  auto stream = Stream::create();
  if (!stream) {
    fail("stream");
    return 1;
  }
  test_bind_errors(*stream);
  test_bind_alias_errors(*stream);
  test_swiglu_epilogue(*stream);
  test_zero_extreme_and_reuse(*stream);
  test_model_bind_missing(*stream);
  if (g_failures != 0) {
    std::cerr << g_failures << " mlp unit failures\n";
    return 1;
  }
  std::cout << "mlp unit ok\n";
  return 0;
}
