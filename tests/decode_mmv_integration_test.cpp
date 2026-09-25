#include "decode_mmv_support.hpp"

#include "compiler/identity.hpp"
#include "compiler/quantization/quantizer.hpp"

#include <cstring>
#include <string>
#include <utility>

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
using qw38::decode_mmv::test::bind_input;
using qw38::decode_mmv::test::bind_output;
using qw38::decode_mmv::test::bind_residual;
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
using qw38::format::bf16_to_fp32;
using qw38::format::fp16_to_fp32;
using qw38::format::fp32_to_bf16_rne;
using qw38::format::pack_bf16_dense_tile_v0;
using qw38::format::pack_cuda_v0;
using qw38::format::store_u16_le;

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
  d.codes.pointer = d_codes->as_bytes();
  bind_input(d, d_x->data());
  if (!packed.scales.empty()) {
    d.scales.pointer = d_scales->as_bytes();
  }
  if (epi == DecodeEpilogue::StoreBf16) {
    auto d_y = DeviceBuffer::allocate(packed.logical_n * 2);
    if (!d_y) {
      fail(std::string(tag) + " alloc");
      return false;
    }
    bind_output(d, d_y->data());
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
    bind_output(d, d_y->data());
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
  bind_residual(d, d_r->data());
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

void candidate_grouped_decode(Stream const& stream) {
  for (auto const& [quantizer, layout] : {
           std::pair{LogicalQuantizerId::Q4G64CandidateV1,
                     PhysicalLayoutId::CudaQ4G64CandidateV1},
           std::pair{LogicalQuantizerId::Q8G32CandidateV1,
                     PhysicalLayoutId::CudaQ8G32CandidateV1}}) {
    constexpr std::uint32_t n = 20;
    constexpr std::uint32_t k = 192;
    auto src = bf16_matrix(n, k, 0.7f);
    auto logical = quantize_bf16(quantizer, n, k, src);
    if (!logical) {
      fail("candidate quantize");
      return;
    }
    auto packed = pack_cuda_v0(quantizer, layout, *logical);
    if (!packed) {
      fail("candidate pack");
      return;
    }
    auto x = bf16_vec(k, 0.2f);
    auto const tag = quantizer == LogicalQuantizerId::Q4G64CandidateV1
                         ? "candidate-q4" : "candidate-q8";
    compare_cuda(*packed, x, DecodeEpilogue::StoreFp32, stream, tag);
    compare_cuda(*packed, x, DecodeEpilogue::ResidualAddFp32, stream, tag);
  }
}

void q4k_mlp(Stream const& stream) {
  constexpr std::uint32_t n = 20, k = 512;
  auto gate = quantize_bf16(LogicalQuantizerId::Q4KCandidateV2, n, k, bf16_matrix(n, k, 0.45f, 1));
  auto up = quantize_bf16(LogicalQuantizerId::Q4KCandidateV2, n, k, bf16_matrix(n, k, 0.55f, 4));
  if (!gate || !up) { fail("q4k MLP quantize"); return; }
  auto pg = pack_cuda_v0(LogicalQuantizerId::Q4KCandidateV2, PhysicalLayoutId::CudaQ4KCandidateV2, *gate);
  auto pu = pack_cuda_v0(LogicalQuantizerId::Q4KCandidateV2, PhysicalLayoutId::CudaQ4KCandidateV2, *up);
  if (!pg || !pu) { fail("q4k MLP pack"); return; }
  auto x = bf16_vec(k, 0.18f);
  compare_cuda(*pg, x, DecodeEpilogue::StoreFp32, stream, "q4k gate FP32");
  compare_cuda(*pu, x, DecodeEpilogue::StoreBf16, stream, "q4k up BF16");
  auto rg = reference_gemv_packed(*pg, x);
  auto ru = reference_gemv_packed(*pu, x);
  auto cg = upload_vec(pg->codes, stream), sg = upload_vec(pg->scales, stream);
  auto cu = upload_vec(pu->codes, stream), su = upload_vec(pu->scales, stream);
  auto dx = upload_vec(x, stream);
  auto dy = DeviceBuffer::allocate(n * 2);
  if (!rg || !ru || !cg || !sg || !cu || !su || !dx || !dy) { fail("q4k MLP setup"); return; }
  DecodeMmvPairedDesc d;
  d.a = desc_from_packed(*pg, DecodeEpilogue::SwigluStoreBf16);
  d.b = desc_from_packed(*pu, DecodeEpilogue::SwigluStoreBf16);
  d.a.codes.pointer = cg->data(); d.a.scales.pointer = sg->data();
  d.b.codes.pointer = cu->data(); d.b.scales.pointer = su->data();
  bind_input(d.a, dx->data()); d.b.input = d.a.input;
  bind_output(d.a, dy->data());
  auto st = launch_decode_mmv_paired(d, stream);
  if (!st) { fail("q4k fused SwiGLU: " + qw38::cuda::error_message(st.error())); return; }
  auto got = download_vec<std::uint16_t>(*dy, n, stream);
  if (!got) { fail("q4k fused SwiGLU download"); return; }
  for (unsigned i = 0; i < n; ++i) {
    float const g = (*rg)[i];
    float const e = std::exp(g >= 0 ? -g : g);
    float const sigmoid = g >= 0 ? 1.0f / (1.0f + e) : e / (1.0f + e);
    (*rg)[i] = (g * sigmoid) * (*ru)[i];
  }
  expect_bf16_close(*got, *rg, "q4k fused gate/up/SwiGLU");

  // Exercise the existing full-width down direct-input dispatch and residual add.
  constexpr std::uint32_t down_k = qw38::cuda::kDecodeMaxK;
  auto down = quantize_bf16(LogicalQuantizerId::Q4KCandidateV2, n, down_k,
                            bf16_matrix(n, down_k, 0.35f, 9));
  if (!down) { fail("q4k down quantize"); return; }
  auto pd = pack_cuda_v0(LogicalQuantizerId::Q4KCandidateV2, PhysicalLayoutId::CudaQ4KCandidateV2, *down);
  if (!pd) { fail("q4k down pack"); return; }
  compare_cuda(*pd, bf16_vec(down_k, 0.2f), DecodeEpilogue::ResidualAddFp32, stream, "q4k down residual");
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
  d.a.codes.pointer = d_cg->as_bytes();
  d.a.scales.pointer = d_sg->as_bytes();
  bind_input(d.a, d_x->data());
  bind_output(d.a, d_yg->data());
  d.b = desc_from_packed(*pu, DecodeEpilogue::StoreBf16);
  d.b.codes.pointer = d_cu->as_bytes();
  d.b.scales.pointer = d_su->as_bytes();
  d.b.input = d.a.input;
  bind_output(d.b, d_yu->data());
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
  d.a.codes.pointer = d_ca->as_bytes();
  bind_input(d.a, d_x->data());
  bind_output(d.a, d_ya->data());
  d.b = desc_from_packed(*pb, DecodeEpilogue::StoreFp32);
  d.b.codes.pointer = d_cb->as_bytes();
  d.b.input = d.a.input;
  bind_output(d.b, d_yb->data());
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

constexpr std::uint32_t kHeadSourceRowPeriod = 21u * 17u;

std::int8_t q8_head_code(std::uint64_t row, std::uint32_t col,
                         std::uint32_t k) {
  return static_cast<std::int8_t>(
      static_cast<int>((row * k + col) % 21u) - 10);
}

std::uint16_t q8_head_scale(std::uint64_t row, std::uint32_t col,
                            std::uint32_t k) {
  auto const groups_per_row = k / 32u;
  auto const group = row * groups_per_row + col / 32u;
  return (group % 17u == 0u) ? std::uint16_t{0x4000}
                             : std::uint16_t{0x3C00};
}

std::uint16_t q8_head_bf16(std::uint64_t row, std::uint32_t col,
                           std::uint32_t k) {
  float const weight =
      static_cast<float>(q8_head_code(row, col, k)) *
      fp16_to_fp32(q8_head_scale(row, col, k));
  return fp32_to_bf16_rne(weight);
}

PackedMatrix make_q8_head_bf16_tiles(std::uint32_t n, std::uint32_t k) {
  PackedMatrix packed;
  packed.layout = PhysicalLayoutId::CudaBf16DenseTileV0;
  packed.quantizer = LogicalQuantizerId::None;
  packed.logical_n = n;
  packed.logical_k = k;
  packed.padded_n = n;
  packed.padded_k = k;
  packed.codes.resize(static_cast<std::size_t>(n) * k * 2u);

  std::vector<std::byte> source_rows(
      static_cast<std::size_t>(kHeadSourceRowPeriod) * k * 2u);
  for (std::uint32_t row = 0; row < kHeadSourceRowPeriod; ++row) {
    for (std::uint32_t col = 0; col < k; ++col) {
      auto const offset = (static_cast<std::size_t>(row) * k + col) * 2u;
      store_u16_le(source_rows.data() + offset, q8_head_bf16(row, col, k));
    }
  }

  auto const tiles_n = n / 8u;
  auto const tiles_k = k / 256u;
  for (std::uint32_t tn = 0; tn < tiles_n; ++tn) {
    for (std::uint32_t tk = 0; tk < tiles_k; ++tk) {
      for (std::uint32_t r = 0; r < 8u; ++r) {
        auto const row = tn * 8u + r;
        auto const source_row = row % kHeadSourceRowPeriod;
        auto const src =
            (static_cast<std::size_t>(source_row) * k + tk * 256u) * 2u;
        auto const dst =
            ((static_cast<std::size_t>(tn) * tiles_k + tk) * 8u + r) * 512u;
        std::memcpy(packed.codes.data() + dst, source_rows.data() + src, 512u);
      }
    }
  }
  return packed;
}

std::vector<float> q8_head_bf16_oracle(
    std::uint32_t n, std::uint32_t k,
    std::span<std::uint16_t const> x) {
  std::vector<float> row_oracle(kHeadSourceRowPeriod);
  for (std::uint32_t row = 0; row < kHeadSourceRowPeriod; ++row) {
    float acc = 0.0f;
    for (std::uint32_t col = 0; col < k; ++col) {
      acc += bf16_to_fp32(q8_head_bf16(row, col, k)) *
             bf16_to_fp32(x[col]);
    }
    row_oracle[row] = acc;
  }
  std::vector<float> out(n);
  for (std::uint32_t row = 0; row < n; ++row) {
    out[row] = row_oracle[row % kHeadSourceRowPeriod];
  }
  return out;
}

void q8_head_bf16_control(std::uint32_t n, std::uint32_t k,
                          std::span<std::uint16_t const> x,
                          Stream const& stream) {
  auto packed = make_q8_head_bf16_tiles(n, k);
  auto reference = q8_head_bf16_oracle(n, k, x);
  expect(packed.logical_n == n && packed.logical_k == k,
         "q8 head bf16-control full logical shape");
  expect(packed.codes.size() == static_cast<std::size_t>(n) * k * 2u,
         "q8 head bf16-control full packed extent");

  auto d_codes = upload_vec(packed.codes, stream);
  std::vector<std::uint16_t> x_owned(x.begin(), x.end());
  auto d_x = upload_vec(x_owned, stream);
  auto d_y = DeviceBuffer::allocate(static_cast<std::uint64_t>(n) * sizeof(float));
  if (!d_codes || !d_x || !d_y) {
    fail("q8 head bf16-control full upload/alloc");
    return;
  }
  auto d = desc_from_packed(packed, DecodeEpilogue::StoreFp32);
  d.codes.pointer = d_codes->as_bytes();
  bind_input(d, d_x->data());
  bind_output(d, d_y->data());
  expect(d.output.dtype == qw38::cuda::DecodeDtype::Fp32,
         "q8 head bf16-control FP32 logits");
  expect(d.output.logical_n == n &&
             d.output.bytes == static_cast<std::uint64_t>(n) * sizeof(float),
         "q8 head bf16-control output extent");

  auto st = launch_decode_mmv(d, stream);
  if (!st) {
    fail("q8 head bf16-control full launch: " +
         qw38::cuda::error_message(st.error()));
    return;
  }
  auto got = download_vec<float>(*d_y, n, stream);
  if (!got) {
    fail("q8 head bf16-control full download");
    return;
  }
  expect_fp32_close(*got, reference, "q8-head-bf16-control-full");
}

void q8_head(Stream const& stream) {
  auto n = static_cast<std::uint32_t>(kVocab);
  auto k = static_cast<std::uint32_t>(kHidden);
  auto x = bf16_vec(k, 0.05f);
  {
    auto logical = make_logical(LogicalQuantizerId::Q8G32V0, n, k, 0, 0x3C00);
    for (std::uint32_t row = 0; row < n; ++row) {
      for (std::uint32_t col = 0; col < k; ++col) {
        logical.codes[static_cast<std::size_t>(row) * k + col] =
            q8_head_code(row, col, k);
      }
      for (std::uint32_t group = 0; group < k / 32u; ++group) {
        logical.scales[static_cast<std::size_t>(row) * (k / 32u) + group] =
            q8_head_scale(row, group * 32u, k);
      }
    }
    auto packed =
        pack_cuda_v0(LogicalQuantizerId::Q8G32V0,
                     PhysicalLayoutId::CudaQ8G32V0, logical);
    if (!packed) {
      fail("q8 head pack: " + qw38::format::error_message(packed.error()));
      return;
    }
    logical = {};
    compare_cuda(*packed, x, DecodeEpilogue::StoreFp32, stream, "q8-head");
  }
  q8_head_bf16_control(n, k, x, stream);
}

}  // namespace

int main() {
  auto stream = Stream::create();
  if (!stream) {
    fail("stream");
    return 1;
  }
  candidate_grouped_decode(*stream);
  q4k_mlp(*stream);

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
