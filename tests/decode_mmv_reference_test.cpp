#include "decode_mmv_support.hpp"

#include "compiler/quantization/quantizer.hpp"

#include <string>

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
using qw38::decode_mmv::test::bf16_matrix;
using qw38::decode_mmv::test::bf16_vec;
using qw38::decode_mmv::test::desc_from_packed;
using qw38::decode_mmv::test::download_vec;
using qw38::decode_mmv::test::expect;
using qw38::decode_mmv::test::expect_bf16_close;
using qw38::decode_mmv::test::expect_fp32_close;
using qw38::decode_mmv::test::fail;
using qw38::decode_mmv::test::g_failures;
using qw38::decode_mmv::test::residual_vec;
using qw38::decode_mmv::test::upload_vec;
using qw38::format::LogicalQuantizerId;
using qw38::format::PackedMatrix;
using qw38::format::PhysicalLayoutId;
using qw38::format::pack_bf16_dense_tile_v0;
using qw38::format::pack_cuda_v0;

namespace {

bool run_packed(PackedMatrix const& packed, std::vector<std::uint16_t> const& x,
                DecodeEpilogue epi, Stream const& stream, std::string_view tag) {
  auto xref = reference_gemv_packed(packed, x);
  if (!xref) {
    fail(std::string(tag) + " cpu: " +
         qw38::compiler::error_message(xref.error()));
    return false;
  }
  auto d_codes = upload_vec(packed.codes, stream);
  std::expected<DeviceBuffer, qw38::cuda::Error> d_scales;
  bool const has_scales = !packed.scales.empty();
  if (has_scales) {
    d_scales = upload_vec(packed.scales, stream);
  }
  auto d_x = upload_vec(x, stream);
  if (!d_codes || !d_x || (has_scales && !d_scales)) {
    fail(std::string(tag) + " upload");
    return false;
  }
  DecodeMmvDesc d = desc_from_packed(packed, epi);
  d.codes = d_codes->as_bytes();
  d.input = static_cast<std::uint16_t const*>(d_x->data());
  if (has_scales) {
    d.scales = d_scales->as_bytes();
  }

  if (epi == DecodeEpilogue::StoreFp32) {
    auto d_y = DeviceBuffer::allocate(packed.logical_n * sizeof(float));
    if (!d_y) {
      fail(std::string(tag) + " y alloc");
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
  if (epi == DecodeEpilogue::StoreBf16) {
    auto d_y = DeviceBuffer::allocate(packed.logical_n * sizeof(std::uint16_t));
    if (!d_y) {
      fail(std::string(tag) + " y alloc");
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
  auto resid0 = residual_vec(static_cast<std::uint32_t>(packed.logical_n), 0.75f);
  auto d_r = upload_vec(resid0, stream);
  if (!d_r) {
    fail(std::string(tag) + " residual upload");
    return false;
  }
  d.output = nullptr;
  d.residual = static_cast<float*>(d_r->data());
  auto st = launch_decode_mmv(d, stream);
  if (!st) {
    fail(std::string(tag) + " launch: " + qw38::cuda::error_message(st.error()));
    return false;
  }
  auto got = download_vec<float>(*d_r, static_cast<std::size_t>(packed.logical_n),
                                 stream);
  if (!got) {
    fail(std::string(tag) + " download");
    return false;
  }
  std::vector<float> want = resid0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    want[i] += (*xref)[i];
  }
  expect_fp32_close(*got, want, tag);
  return true;
}

void quant_family(LogicalQuantizerId qid, PhysicalLayoutId layout, std::uint32_t n,
                  std::uint32_t k, float wscale, Stream const& stream,
                  std::string_view tag) {
  auto src = bf16_matrix(n, k, wscale);
  auto logical = quantize_bf16(qid, n, k, src);
  if (!logical) {
    fail(std::string(tag) + " quantize");
    return;
  }
  auto packed = pack_cuda_v0(qid, layout, *logical);
  if (!packed) {
    fail(std::string(tag) + " pack");
    return;
  }
  auto x = bf16_vec(k, 0.25f);
  run_packed(*packed, x, DecodeEpilogue::StoreFp32, stream, std::string(tag) + " fp32");
  run_packed(*packed, x, DecodeEpilogue::StoreBf16, stream, std::string(tag) + " bf16");
  run_packed(*packed, x, DecodeEpilogue::ResidualAddFp32, stream,
             std::string(tag) + " residual");
}

void bf16_family(std::uint32_t n, std::uint32_t k, float wscale, Stream const& stream,
                 std::string_view tag) {
  auto src = bf16_matrix(n, k, wscale);
  auto packed = pack_bf16_dense_tile_v0(src, n, k);
  if (!packed) {
    fail(std::string(tag) + " pack");
    return;
  }
  auto x = bf16_vec(k, 0.5f);
  run_packed(*packed, x, DecodeEpilogue::StoreFp32, stream, std::string(tag) + " fp32");
  run_packed(*packed, x, DecodeEpilogue::StoreBf16, stream, std::string(tag) + " bf16");
  run_packed(*packed, x, DecodeEpilogue::ResidualAddFp32, stream,
             std::string(tag) + " residual");
}

void paired_q4(Stream const& stream) {
  std::uint32_t const n = 16;
  std::uint32_t const k = 256;
  auto src_a = bf16_matrix(n, k, 0.8f, 1);
  auto src_b = bf16_matrix(n, k, 1.1f, 3);
  auto la = quantize_bf16(LogicalQuantizerId::Q4G64V0, n, k, src_a);
  auto lb = quantize_bf16(LogicalQuantizerId::Q4G64V0, n, k, src_b);
  if (!la || !lb) {
    fail("paired quantize");
    return;
  }
  auto pa = pack_cuda_v0(LogicalQuantizerId::Q4G64V0, PhysicalLayoutId::CudaQ4G64V0, *la);
  auto pb = pack_cuda_v0(LogicalQuantizerId::Q4G64V0, PhysicalLayoutId::CudaQ4G64V0, *lb);
  auto x = bf16_vec(k, 0.3f);
  if (!pa || !pb) {
    fail("paired pack");
    return;
  }
  auto ra = reference_gemv_packed(*pa, x);
  auto rb = reference_gemv_packed(*pb, x);
  auto d_ca = upload_vec(pa->codes, stream);
  auto d_sa = upload_vec(pa->scales, stream);
  auto d_cb = upload_vec(pb->codes, stream);
  auto d_sb = upload_vec(pb->scales, stream);
  auto d_x = upload_vec(x, stream);
  auto d_ya = DeviceBuffer::allocate(n * sizeof(std::uint16_t));
  auto d_yb = DeviceBuffer::allocate(n * sizeof(std::uint16_t));
  if (!ra || !rb || !d_ca || !d_sa || !d_cb || !d_sb || !d_x || !d_ya || !d_yb) {
    fail("paired setup");
    return;
  }
  DecodeMmvPairedDesc d;
  d.a = desc_from_packed(*pa, DecodeEpilogue::StoreBf16);
  d.a.codes = d_ca->as_bytes();
  d.a.scales = d_sa->as_bytes();
  d.a.input = static_cast<std::uint16_t const*>(d_x->data());
  d.a.output = d_ya->data();
  d.codes_b = d_cb->as_bytes();
  d.scales_b = d_sb->as_bytes();
  d.codes_b_bytes = pb->codes.size();
  d.scales_b_bytes = pb->scales.size();
  d.output_b = d_yb->data();
  auto st = launch_decode_mmv_paired(d, stream);
  if (!st) {
    fail("paired launch: " + qw38::cuda::error_message(st.error()));
    return;
  }
  auto ga = download_vec<std::uint16_t>(*d_ya, n, stream);
  auto gb = download_vec<std::uint16_t>(*d_yb, n, stream);
  if (!ga || !gb) {
    fail("paired download");
    return;
  }
  expect_bf16_close(*ga, *ra, "paired a");
  expect_bf16_close(*gb, *rb, "paired b");
}

void grouped_ab(Stream const& stream) {
  std::uint32_t const n = 16;
  std::uint32_t const k = 256;
  auto src_a = bf16_matrix(n, k, 0.4f, 2);
  auto src_b = bf16_matrix(n, k, 0.6f, 5);
  auto pa = pack_bf16_dense_tile_v0(src_a, n, k);
  auto pb = pack_bf16_dense_tile_v0(src_b, n, k);
  auto x = bf16_vec(k, 0.2f);
  if (!pa || !pb) {
    fail("ab pack");
    return;
  }
  auto ra = reference_gemv_packed(*pa, x);
  auto rb = reference_gemv_packed(*pb, x);
  auto d_ca = upload_vec(pa->codes, stream);
  auto d_cb = upload_vec(pb->codes, stream);
  auto d_x = upload_vec(x, stream);
  auto d_ya = DeviceBuffer::allocate(n * sizeof(float));
  auto d_yb = DeviceBuffer::allocate(n * sizeof(float));
  if (!ra || !rb || !d_ca || !d_cb || !d_x || !d_ya || !d_yb) {
    fail("ab setup");
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
    fail("ab launch: " + qw38::cuda::error_message(st.error()));
    return;
  }
  auto ga = download_vec<float>(*d_ya, n, stream);
  auto gb = download_vec<float>(*d_yb, n, stream);
  if (!ga || !gb) {
    fail("ab download");
    return;
  }
  expect_fp32_close(*ga, *ra, "ab a");
  expect_fp32_close(*gb, *rb, "ab b");
}

}  // namespace

int main() {
  auto stream = Stream::create();
  if (!stream) {
    fail("stream");
    return 1;
  }
  quant_family(LogicalQuantizerId::Q4G64V0, PhysicalLayoutId::CudaQ4G64V0, 8, 256,
               1.0f, *stream, "q4-8x256");
  quant_family(LogicalQuantizerId::Q4G64V0, PhysicalLayoutId::CudaQ4G64V0, 16, 512,
               0.35f, *stream, "q4-16x512");
  quant_family(LogicalQuantizerId::Q4G64V0, PhysicalLayoutId::CudaQ4G64V0, 24, 256,
               2.5f, *stream, "q4-24x256");
  quant_family(LogicalQuantizerId::Q8G32V0, PhysicalLayoutId::CudaQ8G32V0, 8, 256,
               4.0f, *stream, "q8-8x256");
  quant_family(LogicalQuantizerId::Q8G32V0, PhysicalLayoutId::CudaQ8G32V0, 16, 512,
               0.9f, *stream, "q8-16x512");
  bf16_family(8, 256, 1.25f, *stream, "bf16-8x256");
  bf16_family(16, 512, 0.55f, *stream, "bf16-16x512");
  paired_q4(*stream);
  grouped_ab(*stream);
  if (g_failures != 0) {
    std::cerr << g_failures << " decode_mmv reference failures\n";
    return 1;
  }
  std::cout << "decode_mmv reference ok\n";
  return 0;
}
