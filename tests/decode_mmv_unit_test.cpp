#include "decode_mmv_support.hpp"

#include "cuda/error.hpp"

#include <array>
#include <limits>
#include <string>

using qw38::cuda::DecodeEpilogue;
using qw38::cuda::DecodeMmvDesc;
using qw38::cuda::DecodeMmvPairedDesc;
using qw38::cuda::DecodeMmvRangeDesc;
using qw38::cuda::DeviceBuffer;
using qw38::cuda::ErrorCode;
using qw38::cuda::Stream;
using qw38::cuda::decode_code_bytes;
using qw38::cuda::decode_pad_n;
using qw38::cuda::decode_pad_k;
using qw38::cuda::decode_scale_bytes;
using qw38::cuda::kDecodeLayoutBf16DenseTileV0;
using qw38::cuda::kDecodeLayoutQ4G64V0;
using qw38::cuda::kDecodeMaxN;
using qw38::cuda::kDecodeQuantizerNone;
using qw38::cuda::kDecodeQuantizerQ4G64V0;
using qw38::cuda::kDecodeQuantizerQ8G32V0;
using qw38::cuda::launch_decode_ab_bf16;
using qw38::cuda::launch_decode_mmv;
using qw38::cuda::launch_decode_mmv_paired;
using qw38::cuda::launch_decode_mmv_ranges;
using qw38::decode_mmv::test::desc_from_packed;
using qw38::decode_mmv::test::bind_input;
using qw38::decode_mmv::test::bind_output;
using qw38::decode_mmv::test::bind_residual;
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
  auto buf = DeviceBuffer::allocate(16384);
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
  d.codes = qw38::cuda::decode_matrix_view(
      buf->data(), qw38::cuda::DecodeDtype::Q4, kDecodeLayoutQ4G64V0,
      8, 256, 8, 256, decode_code_bytes(kDecodeLayoutQ4G64V0, 8, 256), 16);
  d.scales = qw38::cuda::decode_matrix_view(
      static_cast<std::byte*>(buf->data()) + 2048,
      qw38::cuda::DecodeDtype::Fp16, kDecodeLayoutQ4G64V0,
      8, 4, 8, 4, decode_scale_bytes(kDecodeLayoutQ4G64V0, 8, 256), 2);
  bind_input(d, static_cast<std::byte*>(buf->data()) + 2112);
  bind_output(d, static_cast<std::byte*>(buf->data()) + 3072);
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

  d.codes.bytes = 4;
  auto bad_codes = launch_decode_mmv(d, stream);
  expect(!bad_codes && bad_codes.error().code == ErrorCode::InvalidArgument,
         "code length mismatch rejects");
  d.codes.bytes = decode_code_bytes(kDecodeLayoutQ4G64V0, 8, 256);

  expect(decode_pad_n(kDecodeMaxN) == kDecodeMaxN,
         "maximum supported N pads exactly");
  expect(decode_pad_n(kDecodeMaxN + 1u) ==
             static_cast<std::uint64_t>(kDecodeMaxN) + 8u,
         "one-past supported N padding stays wide");
  expect(decode_pad_n(std::numeric_limits<std::uint32_t>::max()) ==
             std::uint64_t{1} << 32,
         "UINT32_MAX padding does not wrap");

  auto zero_tiles = d;
  zero_tiles.n = 0;
  zero_tiles.padded_n = 0;
  auto zero_single = launch_decode_mmv(zero_tiles, stream);
  expect(!zero_single && zero_single.error().code == ErrorCode::InvalidArgument,
         "single zero-tile geometry rejects");

  auto overflow_tiles = d;
  overflow_tiles.n = std::numeric_limits<std::uint32_t>::max();
  overflow_tiles.padded_n = 0;
  auto overflow_single = launch_decode_mmv(overflow_tiles, stream);
  expect(!overflow_single && overflow_single.error().code == ErrorCode::Overflow,
         "single overflowing N rejects before launch");

  std::array<DecodeMmvDesc, 2> ranged{d, zero_tiles};
  auto zero_ranged = launch_decode_mmv_ranges(DecodeMmvRangeDesc{ranged}, stream);
  expect(!zero_ranged && zero_ranged.error().code == ErrorCode::InvalidArgument,
         "ranged zero-tile geometry rejects");
  ranged[1] = overflow_tiles;
  auto overflow_ranged =
      launch_decode_mmv_ranges(DecodeMmvRangeDesc{ranged}, stream);
  expect(!overflow_ranged &&
             overflow_ranged.error().code == ErrorCode::Overflow,
         "ranged overflowing N rejects before launch");

  d.epilogue = DecodeEpilogue::ResidualAddFp32;
  d.output = {};
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
  ab.a.codes = qw38::cuda::decode_matrix_view(
      buf->data(), qw38::cuda::DecodeDtype::Bf16,
      kDecodeLayoutBf16DenseTileV0, 8, 256, 8, 256,
      decode_code_bytes(kDecodeLayoutBf16DenseTileV0, 8, 256), 16);
  bind_input(ab.a, static_cast<std::byte*>(buf->data()) + 2048);
  bind_output(ab.a, static_cast<std::byte*>(buf->data()) + 3072);
  ab.a.epilogue = DecodeEpilogue::StoreBf16;
  ab.b = ab.a;
  ab.b.codes.pointer = static_cast<std::byte*>(buf->data()) + 4096;
  auto ab_wrong = launch_decode_ab_bf16(ab, stream);
  expect(!ab_wrong && ab_wrong.error().code == ErrorCode::InvalidArgument,
         "a/b requires FP32 epilogue");

  auto make_q4 = [&](std::uint32_t n, std::uint32_t k, std::uint64_t off) {
    DecodeMmvDesc x;
    x.layout = kDecodeLayoutQ4G64V0;
    x.quantizer = kDecodeQuantizerQ4G64V0;
    x.n = n;
    x.k = k;
    x.padded_n = qw38::cuda::decode_pad_n(n);
    x.padded_k = decode_pad_k(k);
    x.codes = qw38::cuda::decode_matrix_view(
        static_cast<std::byte*>(buf->data()) + off,
        qw38::cuda::DecodeDtype::Q4, x.layout, n, k, x.padded_n, x.padded_k,
        decode_code_bytes(x.layout, x.padded_n, x.padded_k), 16);
    x.scales = qw38::cuda::decode_matrix_view(
        static_cast<std::byte*>(buf->data()) + off + 2048,
        qw38::cuda::DecodeDtype::Fp16, x.layout, x.padded_n, x.padded_k / 64u,
        x.padded_n, x.padded_k / 64u,
        decode_scale_bytes(x.layout, x.padded_n, x.padded_k), 2);
    bind_input(x, static_cast<std::byte*>(buf->data()) + 8192);
    bind_output(x, static_cast<std::byte*>(buf->data()) + off + 3072);
    return x;
  };
  DecodeMmvPairedDesc shape_mismatch;
  shape_mismatch.a = make_q4(8, 512, 0);
  shape_mismatch.b = make_q4(16, 256, 4096);
  shape_mismatch.b.input =
      qw38::cuda::decode_vector_view(
          shape_mismatch.a.input.pointer, qw38::cuda::DecodeDtype::Bf16,
          qw38::cuda::kDecodeLayoutBf16VectorV0, 256, 512, 2, false);
  auto bad_shape = launch_decode_mmv_paired(shape_mismatch, stream);
  expect(!bad_shape && bad_shape.error().code == ErrorCode::InvalidArgument,
         "equal-byte differently shaped paired B rejects");

  DecodeMmvPairedDesc format_mismatch;
  format_mismatch.a = make_q4(16, 256, 0);
  format_mismatch.b = format_mismatch.a;
  format_mismatch.b.layout = qw38::cuda::kDecodeLayoutQ8G32V0;
  format_mismatch.b.quantizer = kDecodeQuantizerQ8G32V0;
  format_mismatch.b.n = 8;
  format_mismatch.b.padded_n = 8;
  format_mismatch.b.codes = qw38::cuda::decode_matrix_view(
      static_cast<std::byte*>(buf->data()) + 4096,
      qw38::cuda::DecodeDtype::Q8, format_mismatch.b.layout, 8, 256, 8, 256,
      2048, 16);
  format_mismatch.b.scales = qw38::cuda::decode_matrix_view(
      static_cast<std::byte*>(buf->data()) + 6144,
      qw38::cuda::DecodeDtype::Fp16, format_mismatch.b.layout, 8, 8, 8, 8,
      128, 2);
  format_mismatch.b.output = qw38::cuda::decode_vector_view(
      static_cast<std::byte*>(buf->data()) + 7168,
      qw38::cuda::DecodeDtype::Bf16,
      qw38::cuda::kDecodeLayoutBf16VectorV0, 8, 16, 2, true);
  auto bad_format = launch_decode_mmv_paired(format_mismatch, stream);
  expect(!bad_format && bad_format.error().code == ErrorCode::InvalidArgument,
         "equal-byte differently formatted paired B rejects");
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
  d.codes.pointer = d_codes->as_bytes();
  d.scales.pointer = d_scales->as_bytes();
  bind_input(d, d_x->data());
  bind_output(d, d_y->data());
  auto bad_space = d;
  bad_space.input.space = qw38::cuda::DecodeMemorySpace::Host;
  expect(!launch_decode_mmv(bad_space, stream),
         "host input typed view rejects");
  auto bad_dtype = d;
  bad_dtype.output.dtype = qw38::cuda::DecodeDtype::Bf16;
  expect(!launch_decode_mmv(bad_dtype, stream),
         "epilogue output dtype mismatch rejects");
  auto bad_write = d;
  bad_write.codes.writable = true;
  expect(!launch_decode_mmv(bad_write, stream),
         "writable weight typed view rejects");
  auto bad_alignment = d;
  bad_alignment.codes.alignment = 8;
  expect(!launch_decode_mmv(bad_alignment, stream),
         "insufficient weight alignment rejects");
  auto bad_overlap = d;
  bad_overlap.output.pointer = bad_overlap.input.pointer;
  expect(!launch_decode_mmv(bad_overlap, stream),
         "overlapping input/output spans reject");
  auto st = launch_decode_mmv(d, stream);
  expect(static_cast<bool>(st), "pad fp32 launch");
  auto got = download_vec<float>(*d_y, 20, stream);
  if (got) {
    expect_fp32_close(*got, *xref, "pad fp32");
  } else {
    fail("pad fp32 download");
  }

  d.epilogue = DecodeEpilogue::StoreBf16;
  bind_output(d, d_h->data());
  st = launch_decode_mmv(d, stream);
  expect(static_cast<bool>(st), "pad bf16 launch");
  auto got_h = download_vec<std::uint16_t>(*d_h, 20, stream);
  if (got_h) {
    expect_bf16_close(*got_h, *xref, "pad bf16");
  } else {
    fail("pad bf16 download");
  }

  d.epilogue = DecodeEpilogue::ResidualAddFp32;
  bind_residual(d, d_r->data());
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
    d.codes.pointer = d_codes->as_bytes();
    d.scales.pointer = d_scales->as_bytes();
    bind_input(d, d_x->data());
    bind_output(d, d_y->data());
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
    d.codes.pointer = d_codes->as_bytes();
    d.scales.pointer = d_scales->as_bytes();
    bind_input(d, d_x->data());
    bind_output(d, d_y->data());
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
