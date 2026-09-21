#include "decode_mmv_support.hpp"

#include "cuda/error.hpp"

#include <string>

using qw38::cuda::DecodeEpilogue;
using qw38::cuda::DecodeMmvDesc;
using qw38::cuda::DecodeMmvPairedDesc;
using qw38::cuda::DeviceBuffer;
using qw38::cuda::ErrorCode;
using qw38::cuda::Stream;
using qw38::cuda::decode_code_bytes;
using qw38::cuda::decode_pad_k;
using qw38::cuda::decode_scale_bytes;
using qw38::cuda::kDecodeLayoutBf16DenseTileV0;
using qw38::cuda::kDecodeLayoutQ4G64V0;
using qw38::cuda::kDecodeQuantizerNone;
using qw38::cuda::kDecodeQuantizerQ4G64V0;
using qw38::cuda::kDecodeQuantizerQ8G32V0;
using qw38::cuda::launch_decode_ab_bf16;
using qw38::cuda::launch_decode_mmv;
using qw38::cuda::launch_decode_mmv_paired;
using qw38::decode_mmv::test::desc_from_packed;
using qw38::decode_mmv::test::download_vec;
using qw38::decode_mmv::test::expect;
using qw38::decode_mmv::test::expect_bf16_close;
using qw38::decode_mmv::test::expect_fp32_close;
using qw38::decode_mmv::test::fail;
using qw38::decode_mmv::test::g_failures;
using qw38::decode_mmv::test::make_logical;
using qw38::decode_mmv::test::upload_vec;
using qw38::format::LogicalQuantizerId;
using qw38::format::PhysicalLayoutId;
using qw38::format::pack_bf16_dense_tile_v0;
using qw38::format::pack_cuda_v0;

namespace {

void test_launch_validation(Stream const& stream) {
  auto buf = DeviceBuffer::allocate(4096);
  expect(static_cast<bool>(buf), "tiny buffer");
  if (!buf) {
    return;
  }
  DecodeMmvDesc d;
  d.layout = kDecodeLayoutQ4G64V0;
  d.quantizer = kDecodeQuantizerQ8G32V0;
  d.n = 8;
  d.k = 256;
  d.padded_n = 8;
  d.padded_k = 256;
  d.codes = buf->as_bytes();
  d.codes_bytes = decode_code_bytes(kDecodeLayoutQ4G64V0, 8, 256);
  d.scales = buf->as_bytes();
  d.scales_bytes = decode_scale_bytes(kDecodeLayoutQ4G64V0, 8, 256);
  d.input = static_cast<std::uint16_t const*>(buf->data());
  d.output = buf->data();
  auto mismatch = launch_decode_mmv(d, stream);
  expect(!mismatch && mismatch.error().code == ErrorCode::InvalidArgument,
         "layout/quantizer mismatch rejects");

  d.quantizer = kDecodeQuantizerQ4G64V0;
  d.layout = 0x0204;  // cuda_bf16_row_major_v0 is not a decode MMV layout
  auto bad_layout = launch_decode_mmv(d, stream);
  expect(!bad_layout && bad_layout.error().code == ErrorCode::InvalidArgument,
         "unsupported layout rejects");

  d.layout = kDecodeLayoutQ4G64V0;
  Stream empty;
  auto bad_stream = launch_decode_mmv(d, empty);
  expect(!bad_stream && bad_stream.error().code == ErrorCode::InvalidArgument,
         "empty stream rejects");

  d.padded_n = 16;
  auto bad_pad = launch_decode_mmv(d, stream);
  expect(!bad_pad && bad_pad.error().code == ErrorCode::InvalidArgument,
         "padded metadata mismatch rejects");
  d.padded_n = 8;

  d.k = 17409;
  d.padded_k = decode_pad_k(17409);
  auto too_wide = launch_decode_mmv(d, stream);
  expect(!too_wide && too_wide.error().code == ErrorCode::InvalidArgument,
         "K>17408 rejects");
  d.k = 256;
  d.padded_k = 256;

  d.codes_bytes = 4;
  auto bad_codes = launch_decode_mmv(d, stream);
  expect(!bad_codes && bad_codes.error().code == ErrorCode::InvalidArgument,
         "code length mismatch rejects");
  d.codes_bytes = decode_code_bytes(kDecodeLayoutQ4G64V0, 8, 256);

  d.epilogue = DecodeEpilogue::ResidualAddFp32;
  d.residual = nullptr;
  auto bad_res = launch_decode_mmv(d, stream);
  expect(!bad_res && bad_res.error().code == ErrorCode::InvalidArgument,
         "residual-add without residual rejects");

  DecodeMmvPairedDesc ab;
  ab.a.layout = kDecodeLayoutBf16DenseTileV0;
  ab.a.quantizer = kDecodeQuantizerNone;
  ab.a.n = 8;
  ab.a.k = 256;
  ab.a.padded_n = 8;
  ab.a.padded_k = 256;
  ab.a.codes = buf->as_bytes();
  ab.a.codes_bytes = decode_code_bytes(kDecodeLayoutBf16DenseTileV0, 8, 256);
  ab.a.input = static_cast<std::uint16_t const*>(buf->data());
  ab.a.output = buf->data();
  ab.a.epilogue = DecodeEpilogue::StoreBf16;
  ab.codes_b = buf->as_bytes();
  ab.codes_b_bytes = ab.a.codes_bytes;
  ab.output_b = buf->data();
  auto ab_wrong = launch_decode_ab_bf16(ab, stream);
  expect(!ab_wrong && ab_wrong.error().code == ErrorCode::InvalidArgument,
         "a/b requires FP32 epilogue");
}

void test_padding_and_epilogues(Stream const& stream) {
  auto logical = make_logical(LogicalQuantizerId::Q4G64V0, 20, 192, 3, 0x3C00);
  auto packed = pack_cuda_v0(LogicalQuantizerId::Q4G64V0, PhysicalLayoutId::CudaQ4G64V0,
                             logical);
  if (!packed) {
    fail("pad pack: " + qw38::format::error_message(packed.error()));
    return;
  }
  expect(packed->padded_n == 24 && packed->padded_k == 256, "pad metadata 20x192");
  auto x = qw38::decode_mmv::test::bf16_vec(192, 0.25f);
  auto xref = qw38::compiler::reference_gemv_packed(*packed, x);
  if (!xref) {
    fail("pad cpu ref");
    return;
  }

  auto d_codes = upload_vec(packed->codes, stream);
  auto d_scales = upload_vec(packed->scales, stream);
  auto d_x = upload_vec(x, stream);
  auto d_y = DeviceBuffer::allocate(20 * sizeof(float));
  auto d_h = DeviceBuffer::allocate(20 * sizeof(std::uint16_t));
  auto resid0 = qw38::decode_mmv::test::residual_vec(20, 0.5f);
  auto d_r = upload_vec(resid0, stream);
  if (!d_codes || !d_scales || !d_x || !d_y || !d_h || !d_r) {
    fail("pad upload");
    return;
  }

  DecodeMmvDesc d = desc_from_packed(*packed, DecodeEpilogue::StoreFp32);
  d.codes = d_codes->as_bytes();
  d.scales = d_scales->as_bytes();
  d.input = static_cast<std::uint16_t const*>(d_x->data());
  d.output = d_y->data();
  auto st = launch_decode_mmv(d, stream);
  expect(static_cast<bool>(st), "pad fp32 launch");
  auto got = download_vec<float>(*d_y, 20, stream);
  if (got) {
    expect_fp32_close(*got, *xref, "pad fp32");
  } else {
    fail("pad fp32 download");
  }

  d.epilogue = DecodeEpilogue::StoreBf16;
  d.output = d_h->data();
  st = launch_decode_mmv(d, stream);
  expect(static_cast<bool>(st), "pad bf16 launch");
  auto got_h = download_vec<std::uint16_t>(*d_h, 20, stream);
  if (got_h) {
    expect_bf16_close(*got_h, *xref, "pad bf16");
  } else {
    fail("pad bf16 download");
  }

  d.epilogue = DecodeEpilogue::ResidualAddFp32;
  d.output = nullptr;
  d.residual = static_cast<float*>(d_r->data());
  st = launch_decode_mmv(d, stream);
  expect(static_cast<bool>(st), "pad residual launch");
  auto got_r = download_vec<float>(*d_r, 20, stream);
  if (got_r) {
    std::vector<float> want = resid0;
    for (std::uint32_t i = 0; i < 20; ++i) {
      want[i] += (*xref)[i];
    }
    expect_fp32_close(*got_r, want, "pad residual");
  } else {
    fail("pad residual download");
  }
}

void test_zero_and_extreme(Stream const& stream) {
  {
    auto logical = make_logical(LogicalQuantizerId::Q4G64V0, 8, 256, 0, 0);
    auto packed = pack_cuda_v0(LogicalQuantizerId::Q4G64V0, PhysicalLayoutId::CudaQ4G64V0,
                               logical);
    auto x = qw38::decode_mmv::test::bf16_vec(256, 1.0f);
    if (!packed) {
      fail("zero pack");
      return;
    }
    auto d_codes = upload_vec(packed->codes, stream);
    auto d_scales = upload_vec(packed->scales, stream);
    auto d_x = upload_vec(x, stream);
    auto d_y = DeviceBuffer::allocate(8 * sizeof(float));
    if (!d_codes || !d_scales || !d_x || !d_y) {
      fail("zero upload");
      return;
    }
    DecodeMmvDesc d = desc_from_packed(*packed, DecodeEpilogue::StoreFp32);
    d.codes = d_codes->as_bytes();
    d.scales = d_scales->as_bytes();
    d.input = static_cast<std::uint16_t const*>(d_x->data());
    d.output = d_y->data();
    auto st = launch_decode_mmv(d, stream);
    expect(static_cast<bool>(st), "zero launch");
    auto got = download_vec<float>(*d_y, 8, stream);
    if (got) {
      for (float v : *got) {
        expect(v == 0.0f, "zero codes/scales yield zero");
      }
    }
  }
  {
    auto logical =
        make_logical(LogicalQuantizerId::Q8G32V0, 8, 256, 127, 0x7BFF);  // max fp16
    auto packed = pack_cuda_v0(LogicalQuantizerId::Q8G32V0, PhysicalLayoutId::CudaQ8G32V0,
                               logical);
    auto x = qw38::decode_mmv::test::bf16_vec(256, 0.01f);
    if (!packed) {
      fail("extreme pack");
      return;
    }
    auto xref = qw38::compiler::reference_gemv_packed(*packed, x);
    auto d_codes = upload_vec(packed->codes, stream);
    auto d_scales = upload_vec(packed->scales, stream);
    auto d_x = upload_vec(x, stream);
    auto d_y = DeviceBuffer::allocate(8 * sizeof(float));
    if (!packed || !xref || !d_codes || !d_scales || !d_x || !d_y) {
      fail("extreme setup");
      return;
    }
    DecodeMmvDesc d = desc_from_packed(*packed, DecodeEpilogue::StoreFp32);
    d.codes = d_codes->as_bytes();
    d.scales = d_scales->as_bytes();
    d.input = static_cast<std::uint16_t const*>(d_x->data());
    d.output = d_y->data();
    auto st = launch_decode_mmv(d, stream);
    expect(static_cast<bool>(st), "extreme launch");
    auto got = download_vec<float>(*d_y, 8, stream);
    if (got) {
      expect_fp32_close(*got, *xref, "extreme q8");
    }
  }
}

}  // namespace

int main() {
  auto stream = Stream::create();
  if (!stream) {
    fail("stream: " + qw38::cuda::error_message(stream.error()));
    return 1;
  }
  test_launch_validation(*stream);
  test_padding_and_epilogues(*stream);
  test_zero_and_extreme(*stream);
  if (g_failures != 0) {
    std::cerr << g_failures << " decode_mmv unit failures\n";
    return 1;
  }
  std::cout << "decode_mmv unit ok\n";
  return 0;
}
