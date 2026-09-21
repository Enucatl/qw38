#include "decode_mmv_support.hpp"

#include "compiler/identity.hpp"
#include "compiler/quantization/quantizer.hpp"

#include <string>

using qw38::compiler::kHeadDim;
using qw38::compiler::kHidden;
using qw38::compiler::kIntermediate;
using qw38::compiler::kKvHeads;
using qw38::compiler::kLinearValueHeads;
using qw38::compiler::kQkvWidth;
using qw38::compiler::kQueryHeads;
using qw38::compiler::kVocab;
using qw38::compiler::kZWidth;
using qw38::compiler::quantize_bf16;
using qw38::compiler::reference_gemv_packed;
using qw38::cuda::DecodeEpilogue;
using qw38::cuda::DecodeMmvDesc;
using qw38::cuda::DecodeMmvPairedDesc;
using qw38::cuda::DeviceBuffer;
using qw38::cuda::Stream;
using qw38::cuda::launch_decode_ab_bf16;
using qw38::cuda::launch_decode_mmv;
using qw38::cuda::launch_decode_mmv_paired;
using qw38::decode_mmv::test::download_vec;
using qw38::decode_mmv::test::upload_vec;
using qw38::decode_mmv::test::bf16_matrix;
using qw38::decode_mmv::test::bf16_vec;
using qw38::decode_mmv::test::desc_from_packed;
using qw38::decode_mmv::test::expect;
using qw38::decode_mmv::test::expect_bf16_close;
using qw38::decode_mmv::test::expect_fp32_close;
using qw38::decode_mmv::test::fail;
using qw38::decode_mmv::test::g_failures;
using qw38::decode_mmv::test::make_logical;
using qw38::decode_mmv::test::residual_vec;
using qw38::format::LogicalQuantizerId;
using qw38::format::PackedMatrix;
using qw38::format::PhysicalLayoutId;
using qw38::format::pack_bf16_dense_tile_v0;
using qw38::format::pack_cuda_v0;

namespace {

bool compare_cuda(PackedMatrix const& packed, std::vector<std::uint16_t> const& x,
                  DecodeEpilogue epi, Stream const& stream, std::string_view tag) {
  auto xref = reference_gemv_packed(packed, x);
  if (!xref) {
    fail(std::string(tag) + " cpu: " +
         qw38::compiler::error_message(xref.error()));
    return false;
  }
  auto d_codes = upload_vec(packed.codes, stream);
  std::expected<DeviceBuffer, qw38::cuda::Error> d_scales;
  if (!packed.scales.empty()) {
    d_scales = upload_vec(packed.scales, stream);
  }
  auto d_x = upload_vec(x, stream);
  if (!d_codes || !d_x || (!packed.scales.empty() && !d_scales)) {
    fail(std::string(tag) + " upload");
    return false;
  }
  DecodeMmvDesc d = desc_from_packed(packed, epi);
  d.codes = d_codes->as_bytes();
  d.input = static_cast<std::uint16_t const*>(d_x->data());
  if (!packed.scales.empty()) {
    d.scales = d_scales->as_bytes();
  }
  if (epi == DecodeEpilogue::StoreBf16) {
    auto d_y = DeviceBuffer::allocate(packed.logical_n * 2);
    if (!d_y) {
      fail(std::string(tag) + " alloc");
      return false;
    }
    d.output = d_y->data();
    auto st = launch_decode_mmv(d, stream);
    if (!st) {
      fail(std::string(tag) + " launch: " + qw38::cuda::error_message(st.error()));
      return false;
    }
    auto got = download_vec<std::uint16_t>(
        *d_y, static_cast<std::size_t>(packed.logical_n), stream);
    if (!got) {
      fail(std::string(tag) + " download");
      return false;
    }
    expect_bf16_close(*got, *xref, tag);
    return true;
  }
  if (epi == DecodeEpilogue::StoreFp32) {
    auto d_y = DeviceBuffer::allocate(packed.logical_n * 4);
    if (!d_y) {
      fail(std::string(tag) + " alloc");
      return false;
    }
    d.output = d_y->data();
    auto st = launch_decode_mmv(d, stream);
    if (!st) {
      fail(std::string(tag) + " launch: " + qw38::cuda::error_message(st.error()));
      return false;
    }
    auto got = download_vec<float>(*d_y, static_cast<std::size_t>(packed.logical_n),
                                   stream);
    if (!got) {
      fail(std::string(tag) + " download");
      return false;
    }
    expect_fp32_close(*got, *xref, tag);
    return true;
  }
  auto resid0 = residual_vec(static_cast<std::uint32_t>(packed.logical_n), 0.15f);
  auto d_r = upload_vec(resid0, stream);
  if (!d_r) {
    fail(std::string(tag) + " residual upload");
    return false;
  }
  d.residual = static_cast<float*>(d_r->data());
  auto st = launch_decode_mmv(d, stream);
  if (!st) {
    fail(std::string(tag) + " launch: " + qw38::cuda::error_message(st.error()));
    return false;
  }
  auto got =
      download_vec<float>(*d_r, static_cast<std::size_t>(packed.logical_n), stream);
  if (!got) {
    fail(std::string(tag) + " download");
    return false;
  }
  for (std::size_t i = 0; i < resid0.size(); ++i) {
    resid0[i] += (*xref)[i];
  }
  expect_fp32_close(*got, resid0, tag);
  return true;
}

void q4_and_bf16_control(std::uint32_t n, std::uint32_t k, DecodeEpilogue epi,
                         Stream const& stream, std::string_view tag) {
  auto src = bf16_matrix(n, k, 0.7f);
  auto x = bf16_vec(k, 0.2f);
  auto bf16_packed = pack_bf16_dense_tile_v0(src, n, k);
  if (!bf16_packed) {
    fail(std::string(tag) + " bf16 pack");
    return;
  }
  compare_cuda(*bf16_packed, x, epi, stream, std::string(tag) + " bf16-control");

  auto logical = quantize_bf16(LogicalQuantizerId::Q4G64V0, n, k, src);
  if (!logical) {
    fail(std::string(tag) + " quantize");
    return;
  }
  auto q4 = pack_cuda_v0(LogicalQuantizerId::Q4G64V0, PhysicalLayoutId::CudaQ4G64V0,
                         *logical);
  if (!q4) {
    fail(std::string(tag) + " q4 pack");
    return;
  }
  compare_cuda(*q4, x, epi, stream, std::string(tag) + " q4");
}

void mlp_paired(Stream const& stream) {
  auto n = static_cast<std::uint32_t>(kIntermediate);
  auto k = static_cast<std::uint32_t>(kHidden);
  auto src_g = bf16_matrix(n, k, 0.45f, 1);
  auto src_u = bf16_matrix(n, k, 0.55f, 4);
  auto x = bf16_vec(k, 0.18f);
  auto lg = quantize_bf16(LogicalQuantizerId::Q4G64V0, n, k, src_g);
  auto lu = quantize_bf16(LogicalQuantizerId::Q4G64V0, n, k, src_u);
  if (!lg || !lu) {
    fail("mlp paired quantize");
    return;
  }
  auto pg = pack_cuda_v0(LogicalQuantizerId::Q4G64V0, PhysicalLayoutId::CudaQ4G64V0, *lg);
  auto pu = pack_cuda_v0(LogicalQuantizerId::Q4G64V0, PhysicalLayoutId::CudaQ4G64V0, *lu);
  if (!pg || !pu) {
    fail("mlp paired pack");
    return;
  }
  auto rg = reference_gemv_packed(*pg, x);
  auto ru = reference_gemv_packed(*pu, x);
  auto d_cg = upload_vec(pg->codes, stream);
  auto d_sg = upload_vec(pg->scales, stream);
  auto d_cu = upload_vec(pu->codes, stream);
  auto d_su = upload_vec(pu->scales, stream);
  auto d_x = upload_vec(x, stream);
  auto d_yg = DeviceBuffer::allocate(static_cast<std::uint64_t>(n) * 2);
  auto d_yu = DeviceBuffer::allocate(static_cast<std::uint64_t>(n) * 2);
  if (!rg || !ru || !d_cg || !d_sg || !d_cu || !d_su || !d_x || !d_yg || !d_yu) {
    fail("mlp paired setup");
    return;
  }
  DecodeMmvPairedDesc d;
  d.a = desc_from_packed(*pg, DecodeEpilogue::StoreBf16);
  d.a.codes = d_cg->as_bytes();
  d.a.scales = d_sg->as_bytes();
  d.a.input = static_cast<std::uint16_t const*>(d_x->data());
  d.a.output = d_yg->data();
  d.codes_b = d_cu->as_bytes();
  d.scales_b = d_su->as_bytes();
  d.codes_b_bytes = pu->codes.size();
  d.scales_b_bytes = pu->scales.size();
  d.output_b = d_yu->data();
  auto st = launch_decode_mmv_paired(d, stream);
  if (!st) {
    fail("mlp paired launch: " + qw38::cuda::error_message(st.error()));
    return;
  }
  auto gg = download_vec<std::uint16_t>(*d_yg, n, stream);
  auto gu = download_vec<std::uint16_t>(*d_yu, n, stream);
  if (!gg || !gu) {
    fail("mlp paired download");
    return;
  }
  expect_bf16_close(*gg, *rg, "mlp gate");
  expect_bf16_close(*gu, *ru, "mlp up");
}

void gdn_ab(Stream const& stream) {
  auto n = static_cast<std::uint32_t>(kLinearValueHeads);
  auto k = static_cast<std::uint32_t>(kHidden);
  auto src_a = bf16_matrix(n, k, 0.3f, 7);
  auto src_b = bf16_matrix(n, k, 0.35f, 9);
  auto x = bf16_vec(k, 0.12f);
  auto pa = pack_bf16_dense_tile_v0(src_a, n, k);
  auto pb = pack_bf16_dense_tile_v0(src_b, n, k);
  if (!pa || !pb) {
    fail("gdn a/b pack");
    return;
  }
  auto ra = reference_gemv_packed(*pa, x);
  auto rb = reference_gemv_packed(*pb, x);
  auto d_ca = upload_vec(pa->codes, stream);
  auto d_cb = upload_vec(pb->codes, stream);
  auto d_x = upload_vec(x, stream);
  auto d_ya = DeviceBuffer::allocate(n * 4);
  auto d_yb = DeviceBuffer::allocate(n * 4);
  if (!ra || !rb || !d_ca || !d_cb || !d_x || !d_ya || !d_yb) {
    fail("gdn a/b setup");
    return;
  }
  DecodeMmvPairedDesc d;
  d.a = desc_from_packed(*pa, DecodeEpilogue::StoreFp32);
  d.a.codes = d_ca->as_bytes();
  d.a.input = static_cast<std::uint16_t const*>(d_x->data());
  d.a.output = d_ya->data();
  d.codes_b = d_cb->as_bytes();
  d.codes_b_bytes = pb->codes.size();
  d.output_b = d_yb->data();
  auto st = launch_decode_ab_bf16(d, stream);
  if (!st) {
    fail("gdn a/b launch: " + qw38::cuda::error_message(st.error()));
    return;
  }
  auto ga = download_vec<float>(*d_ya, n, stream);
  auto gb = download_vec<float>(*d_yb, n, stream);
  if (!ga || !gb) {
    fail("gdn a/b download");
    return;
  }
  expect_fp32_close(*ga, *ra, "gdn a");
  expect_fp32_close(*gb, *rb, "gdn b");
}

void q8_head(Stream const& stream) {
  auto n = static_cast<std::uint32_t>(kVocab);
  auto k = static_cast<std::uint32_t>(kHidden);
  auto logical = make_logical(LogicalQuantizerId::Q8G32V0, n, k, 0, 0x3C00);
  for (std::uint64_t i = 0; i < logical.codes.size(); ++i) {
    int const v = static_cast<int>(i % 21) - 10;
    logical.codes[static_cast<std::size_t>(i)] = static_cast<std::int8_t>(v);
  }
  for (std::uint64_t i = 0; i < logical.scales.size(); ++i) {
    logical.scales[static_cast<std::size_t>(i)] =
        (i % 17 == 0) ? std::uint16_t{0x4000} : std::uint16_t{0x3C00};
  }
  auto packed = pack_cuda_v0(LogicalQuantizerId::Q8G32V0, PhysicalLayoutId::CudaQ8G32V0,
                             logical);
  if (!packed) {
    fail("q8 head pack: " + qw38::format::error_message(packed.error()));
    return;
  }
  auto x = bf16_vec(k, 0.05f);
  compare_cuda(*packed, x, DecodeEpilogue::StoreFp32, stream, "q8-head");

  auto slice_src = bf16_matrix(256, k, 0.4f);
  auto slice = pack_bf16_dense_tile_v0(slice_src, 256, k);
  if (!slice) {
    fail("q8 head bf16-control pack");
    return;
  }
  compare_cuda(*slice, x, DecodeEpilogue::StoreFp32, stream, "q8-head-bf16-control-256");
}

}  // namespace

int main() {
  auto stream = Stream::create();
  if (!stream) {
    fail("stream");
    return 1;
  }

  auto qkv_n = static_cast<std::uint32_t>(kQkvWidth);
  auto z_n = static_cast<std::uint32_t>(kZWidth);
  auto hidden = static_cast<std::uint32_t>(kHidden);
  auto qg_n = static_cast<std::uint32_t>(2 * kQueryHeads * kHeadDim);
  auto kv_n = static_cast<std::uint32_t>(kKvHeads * kHeadDim);
  auto ffn = static_cast<std::uint32_t>(kIntermediate);

  q4_and_bf16_control(qkv_n, hidden, DecodeEpilogue::StoreBf16, *stream, "gdn-qkv");
  q4_and_bf16_control(z_n, hidden, DecodeEpilogue::StoreBf16, *stream, "gdn-z");
  q4_and_bf16_control(hidden, z_n, DecodeEpilogue::ResidualAddFp32, *stream, "gdn-out");
  gdn_ab(*stream);

  q4_and_bf16_control(qg_n, hidden, DecodeEpilogue::StoreBf16, *stream, "attn-qg");
  q4_and_bf16_control(kv_n, hidden, DecodeEpilogue::StoreBf16, *stream, "attn-k");
  q4_and_bf16_control(hidden, z_n, DecodeEpilogue::ResidualAddFp32, *stream, "attn-o");

  mlp_paired(*stream);
  q4_and_bf16_control(hidden, ffn, DecodeEpilogue::ResidualAddFp32, *stream, "mlp-down");

  q8_head(*stream);

  if (g_failures != 0) {
    std::cerr << g_failures << " decode_mmv integration failures\n";
    return 1;
  }
  std::cout << "decode_mmv integration ok\n";
  return 0;
}
