#include "compiler/compiler.hpp"
#include "format/format.hpp"
#include "format_reader_support.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using qw38::compiler::compile_synthetic;
using qw38::compiler::current_peak_rss_bytes;
using qw38::compiler::dequantize_to_bf16;
using qw38::compiler::expand_identity_table;
using qw38::compiler::kProductionCompilerIdent;
using qw38::compiler::open_checkpoint;
using qw38::compiler::quantize_bf16;
using qw38::compiler::quantize_group_into;
using qw38::compiler::select_weight_format;
using qw38::compiler::SyntheticTensor;
using qw38::compiler::TensorFamily;
using qw38::compiler::verify_quantized_tensor;
using qw38::compiler::WeightFormatPolicy;
using qw38::format::Artifact;
using qw38::format::expected_payload_bytes;
using qw38::format::expected_scale_bytes;
using qw38::format::Hash256;
using qw38::format::kSpanAlignment;
using qw38::format::LogicalQuantizerId;
using qw38::format::pack_cuda_v0;
using qw38::format::PhysicalLayoutId;
using qw38::format::sha256;
using qw38::format::StorageClass;
using qw38::format::test::ScratchDir;
using qw38::format::unpack_cuda_v0;

#ifndef QW38_SOURCE_DIR
#define QW38_SOURCE_DIR "."
#endif

namespace {

int g_failures = 0;

void fail(std::string_view what) {
  std::cerr << "FAIL: " << what << '\n';
  ++g_failures;
}

void expect(bool cond, std::string_view what) {
  if (!cond) {
    fail(what);
  }
}

Hash256 hash_seed(std::uint8_t seed) {
  Hash256 h{};
  for (std::size_t i = 0; i < h.bytes.size(); ++i) {
    h.bytes[i] = static_cast<std::uint8_t>(seed + i);
  }
  return h;
}

std::vector<std::byte> pattern(std::uint64_t elems, std::uint16_t seed) {
  std::vector<std::byte> out(elems * 2);
  for (std::uint64_t i = 0; i < elems; ++i) {
    std::uint16_t bits =
        static_cast<std::uint16_t>(0x3C00 + ((seed + i) & 0x007F));
    std::memcpy(out.data() + i * 2, &bits, 2);
  }
  return out;
}

qw38::compiler::ExpectedTensor find_expected(TensorFamily family) {
  for (auto const& e : expand_identity_table()) {
    if (e.family == family) {
      return e;
    }
  }
  return {};
}

SyntheticTensor small_matrix(qw38::compiler::ExpectedTensor exp, std::uint64_t n,
                             std::uint64_t k, std::uint16_t seed) {
  exp.shape.rank = 2;
  exp.shape.dims = {n, k, 0};
  SyntheticTensor t{};
  t.expected = std::move(exp);
  t.bytes = pattern(n * k, seed);
  return t;
}

SyntheticTensor synth_from(qw38::compiler::ExpectedTensor exp,
                           std::uint16_t seed) {
  std::uint64_t n = 1;
  for (std::uint8_t i = 0; i < exp.shape.rank; ++i) {
    n *= exp.shape.dims[i];
  }
  SyntheticTensor t{};
  t.expected = std::move(exp);
  t.bytes = pattern(n, seed);
  return t;
}

std::vector<std::byte> take_tile(std::span<std::byte const> src, std::uint64_t k,
                                 std::uint64_t rows, std::uint64_t cols) {
  std::vector<std::byte> out(static_cast<std::size_t>(rows * cols * 2));
  for (std::uint64_t r = 0; r < rows; ++r) {
    std::memcpy(out.data() + r * cols * 2, src.data() + r * k * 2, cols * 2);
  }
  return out;
}

}  // namespace

int main() {
  expect(select_weight_format(TensorFamily::LmHead,
                              PhysicalLayoutId::CudaBf16DenseTileV0,
                              WeightFormatPolicy::IdentityBf16)
                 .quantizer == LogicalQuantizerId::None,
         "identity keeps lm_head unquantized");
  expect(select_weight_format(TensorFamily::LmHead,
                              PhysicalLayoutId::CudaBf16DenseTileV0,
                              WeightFormatPolicy::ProductionV0)
                 .quantizer == LogicalQuantizerId::Q8G32V0,
         "production lm_head is Q8G32");
  expect(select_weight_format(TensorFamily::LinearAttnInProjQkv,
                              PhysicalLayoutId::CudaBf16DenseTileV0,
                              WeightFormatPolicy::ProductionV0)
                 .quantizer == LogicalQuantizerId::Q4G64V0,
         "production qkv is Q4G64");
  expect(select_weight_format(TensorFamily::LinearAttnInProjA,
                              PhysicalLayoutId::CudaBf16DenseTileV0,
                              WeightFormatPolicy::ProductionV0)
                 .quantizer == LogicalQuantizerId::None,
         "GDN a/b stay BF16");
  expect(select_weight_format(TensorFamily::Embed,
                              PhysicalLayoutId::CudaBf16RowMajorV0,
                              WeightFormatPolicy::ProductionV0)
                 .layout == PhysicalLayoutId::CudaBf16RowMajorV0,
         "embed family assignment unchanged");

  {
    constexpr std::uint64_t kLargeN = 1024;
    constexpr std::uint64_t kLargeK = 4096;
    constexpr std::uint64_t kElements = kLargeN * kLargeK;
    std::vector<std::byte> source(static_cast<std::size_t>(kElements * 2));
    for (std::uint64_t i = 0; i < kElements; ++i) {
      source[static_cast<std::size_t>(i * 2)] = std::byte{0x80};
      source[static_cast<std::size_t>(i * 2 + 1)] = std::byte{0x3F};
    }
    std::array<float, qw38::format::kQ8GroupSize> group{};
    group.fill(1.0f);
    std::array<std::int8_t, qw38::format::kQ8GroupSize> codes{};
    auto scale =
        quantize_group_into(LogicalQuantizerId::Q8G32V0, group, codes);
    expect(static_cast<bool>(scale), "large verification fixture quantizes");
    std::vector<std::byte> payload(static_cast<std::size_t>(kElements));
    std::fill(payload.begin(), payload.end(),
              static_cast<std::byte>(static_cast<std::uint8_t>(codes[0])));
    std::vector<std::byte> scales(
        static_cast<std::size_t>(kElements / qw38::format::kQ8GroupSize * 2));
    for (std::size_t i = 0; i < scales.size(); i += 2) {
      qw38::format::store_u16_le(scales.data() + i, *scale);
    }

    auto const rss_before = current_peak_rss_bytes();
    auto verified = verify_quantized_tensor(
        "large_q8", LogicalQuantizerId::Q8G32V0,
        PhysicalLayoutId::CudaQ8G32V0, kLargeN, kLargeK, source, payload,
        scales);
    auto const rss_after = current_peak_rss_bytes();
    constexpr std::uint64_t kVerificationRssBudget = 8ULL << 20;
    expect(static_cast<bool>(verified), "large Q8 tensor verifies");
    expect(rss_before != 0 && rss_after >= rss_before &&
               rss_after - rss_before <= kVerificationRssBudget,
           "large verification peak RSS stays within 8 MiB");

    payload.front() ^= std::byte{1};
    auto corrupted = verify_quantized_tensor(
        "large_q8", LogicalQuantizerId::Q8G32V0,
        PhysicalLayoutId::CudaQ8G32V0, kLargeN, kLargeK, source, payload,
        scales);
    expect(!corrupted, "incremental verification rejects corrupt payload");
  }

  ScratchDir dir{"qw38-quant-int"};
  auto embed = small_matrix(find_expected(TensorFamily::Embed), 8, 8, 1);
  embed.expected.layout = PhysicalLayoutId::CudaBf16RowMajorV0;
  auto head = small_matrix(find_expected(TensorFamily::LmHead), 8, 256, 2);
  auto dense = small_matrix(find_expected(TensorFamily::LinearAttnInProjQkv), 8,
                            256, 3);
  auto conv = synth_from(find_expected(TensorFamily::LinearAttnConv1d), 4);
  conv.expected.shape = {.rank = 3, .dims = {32, 1, 4}};
  conv.bytes = pattern(32 * 4, 4);
  auto vec = synth_from(find_expected(TensorFamily::LinearAttnALog), 5);
  std::vector<SyntheticTensor> fixture{embed, head, dense, conv, vec};

  qw38::format::CompilerRevision rev{.ident = kProductionCompilerIdent,
                                     .major = 0,
                                     .minor = 1,
                                     .patch = 0};
  auto dest = dir.path() / "mixed.qw38";
  auto compiled =
      compile_synthetic(dest, hash_seed(0x11), hash_seed(0x22), hash_seed(0x33),
                        rev, fixture, WeightFormatPolicy::ProductionV0);
  if (!compiled) {
    fail(qw38::compiler::error_message(compiled.error()));
    std::cerr << g_failures << " failures\n";
    return 1;
  }

  auto art = Artifact::open(dest);
  if (!art) {
    fail(qw38::format::error_message(art.error()));
    std::cerr << g_failures << " failures\n";
    return 1;
  }
  expect(art->compiler().ident == kProductionCompilerIdent, "production ident");
  expect(art->compiler().major == 0 && art->compiler().minor == 1, "compiler version");

  auto const* q4 = art->find_tensor(dense.expected.name);
  auto const* q8 = art->find_tensor(head.expected.name);
  auto const* emb = art->find_tensor(embed.expected.name);
  expect(q4 && q4->quantizer == LogicalQuantizerId::Q4G64V0 &&
             q4->layout == PhysicalLayoutId::CudaQ4G64V0 &&
             q4->storage == StorageClass::Int4Grouped,
         "Q4 tensor versions");
  expect(q8 && q8->quantizer == LogicalQuantizerId::Q8G32V0 &&
             q8->layout == PhysicalLayoutId::CudaQ8G32V0 &&
             q8->storage == StorageClass::Int8Grouped,
         "Q8 tensor versions");
  expect(emb && emb->quantizer == LogicalQuantizerId::None &&
             emb->layout == PhysicalLayoutId::CudaBf16RowMajorV0,
         "identity embed in mixed artifact");
  expect(q4 && q4->payload.offset % kSpanAlignment == 0 &&
             q4->scales.offset % kSpanAlignment == 0,
         "256-byte bases");

  if (q4 && q8 && emb) {
    auto q4n = expected_payload_bytes(*q4);
    auto q4s = expected_scale_bytes(*q4);
    auto q8n = expected_payload_bytes(*q8);
    auto q8s = expected_scale_bytes(*q8);
    auto en = expected_payload_bytes(*emb);
    auto q4p = art->payload(dense.expected.name);
    auto q4sc = art->scales(dense.expected.name);
    auto q8p = art->payload(head.expected.name);
    auto q8sc = art->scales(head.expected.name);
    auto ep = art->payload(embed.expected.name);
    expect(q4n && q4p && q4p->size() == *q4n, "Q4 payload byte count");
    expect(q4s && q4sc && q4sc->size() == *q4s, "Q4 scale byte count");
    expect(q8n && q8p && q8p->size() == *q8n, "Q8 payload byte count");
    expect(q8s && q8sc && q8sc->size() == *q8s, "Q8 scale byte count");
    expect(en && ep && ep->size() == *en &&
               std::equal(ep->begin(), ep->end(), embed.bytes.begin()),
           "embed identity bytes");

    bool saw_payload_hash = false;
    bool saw_scale_hash = false;
    for (auto const& rec : art->schema().integrity) {
      if (rec.kind == qw38::format::IntegrityKind::Sha256PayloadSpan &&
          rec.tensor_id == q4->tensor_id && q4p) {
        expect(rec.digest == sha256(*q4p), "Q4 payload hash");
        saw_payload_hash = true;
      }
      if (rec.kind == qw38::format::IntegrityKind::Sha256ScaleSpan &&
          rec.tensor_id == q4->tensor_id && q4sc) {
        expect(rec.digest == sha256(*q4sc), "Q4 scale hash");
        saw_scale_hash = true;
      }
    }
    expect(saw_payload_hash && saw_scale_hash, "integrity records present");

    if (q4p && q4sc) {
      auto want = quantize_bf16(LogicalQuantizerId::Q4G64V0, 8, 256, dense.bytes);
      auto packed = pack_cuda_v0(LogicalQuantizerId::Q4G64V0,
                                 PhysicalLayoutId::CudaQ4G64V0, *want);
      expect(static_cast<bool>(want) && static_cast<bool>(packed) &&
                 packed->codes.size() == q4p->size() &&
                 std::equal(q4p->begin(), q4p->end(), packed->codes.begin()) &&
                 std::equal(q4sc->begin(), q4sc->end(), packed->scales.begin()),
             "Q4 reconstructed codes/scales");
      expect(static_cast<bool>(verify_quantized_tensor(
                 dense.expected.name, LogicalQuantizerId::Q4G64V0,
                 PhysicalLayoutId::CudaQ4G64V0, 8, 256, dense.bytes, *q4p,
                 *q4sc)),
             "Q4 incremental verification");
      auto unpacked = unpack_cuda_v0(LogicalQuantizerId::Q4G64V0,
                                     PhysicalLayoutId::CudaQ4G64V0, 8, 256,
                                     *q4p, *q4sc);
      expect(static_cast<bool>(unpacked) && unpacked->codes == want->codes,
             "Q4 unpack matches logical quantizer");
    }
    if (q8p && q8sc) {
      auto want = quantize_bf16(LogicalQuantizerId::Q8G32V0, 8, 256, head.bytes);
      auto packed = pack_cuda_v0(LogicalQuantizerId::Q8G32V0,
                                 PhysicalLayoutId::CudaQ8G32V0, *want);
      expect(static_cast<bool>(want) && static_cast<bool>(packed) &&
                 std::equal(q8p->begin(), q8p->end(), packed->codes.begin()) &&
                 std::equal(q8sc->begin(), q8sc->end(), packed->scales.begin()),
             "Q8 reconstructed codes/scales");
      expect(static_cast<bool>(verify_quantized_tensor(
                 head.expected.name, LogicalQuantizerId::Q8G32V0,
                 PhysicalLayoutId::CudaQ8G32V0, 8, 256, head.bytes, *q8p,
                 *q8sc)),
             "Q8 incremental verification");
    }
  }

  auto dest_id = dir.path() / "identity.qw38";
  qw38::format::CompilerRevision idrev{.ident = qw38::compiler::kCompilerIdent,
                                       .major = 0,
                                       .minor = 1,
                                       .patch = 0};
  auto ident = compile_synthetic(dest_id, hash_seed(0x11), hash_seed(0x22),
                                 hash_seed(0x33), idrev, fixture);
  if (!ident) {
    fail(qw38::compiler::error_message(ident.error()));
  } else {
    auto art_id = Artifact::open(dest_id);
    auto const* dense_id =
        art_id ? art_id->find_tensor(dense.expected.name) : nullptr;
    expect(dense_id && dense_id->quantizer == LogicalQuantizerId::None &&
               dense_id->layout == PhysicalLayoutId::CudaBf16DenseTileV0,
           "identity path still BF16 dense tile");
  }

  std::filesystem::path authority =
      std::filesystem::path(QW38_SOURCE_DIR) /
      ".cache/authorities/qwen3.8-27b-transformers";
  if (std::filesystem::exists(authority / "config.json") &&
      std::filesystem::exists(authority / "model.safetensors.index.json")) {
    auto ckpt = open_checkpoint(authority);
    if (!ckpt) {
      fail(qw38::compiler::error_message(ckpt.error()));
    } else {
      auto load_family = [&](TensorFamily family) -> std::vector<std::byte> {
        for (auto const& item : ckpt->classified.included) {
          if (item.expected.family != family) {
            continue;
          }
          auto mapped = qw38::compiler::MappedShard::open(
              ckpt->shards.at(item.source.shard).path);
          if (!mapped) {
            fail(qw38::compiler::error_message(mapped.error()));
            return {};
          }
          auto bytes = mapped->tensor_bytes(item.source);
          if (!bytes) {
            fail(qw38::compiler::error_message(bytes.error()));
            return {};
          }
          return std::vector<std::byte>(bytes->begin(), bytes->end());
        }
        return {};
      };
      auto qkv = load_family(TensorFamily::LinearAttnInProjQkv);
      auto head_src = load_family(TensorFamily::LmHead);
      expect(!qkv.empty() && !head_src.empty(), "real projection samples");
      if (!qkv.empty() && !head_src.empty()) {
        auto qkv_tile = take_tile(qkv, 5120, 8, 256);
        auto head_tile = take_tile(head_src, 5120, 8, 256);
        auto q4 = quantize_bf16(LogicalQuantizerId::Q4G64V0, 8, 256, qkv_tile);
        auto q8 = quantize_bf16(LogicalQuantizerId::Q8G32V0, 8, 256, head_tile);
        if (!q4 || !q8) {
          fail("real sample quantize");
        } else {
          auto p4 = pack_cuda_v0(LogicalQuantizerId::Q4G64V0,
                                 PhysicalLayoutId::CudaQ4G64V0, *q4);
          auto p8 = pack_cuda_v0(LogicalQuantizerId::Q8G32V0,
                                 PhysicalLayoutId::CudaQ8G32V0, *q8);
          auto u4 = unpack_cuda_v0(LogicalQuantizerId::Q4G64V0,
                                   PhysicalLayoutId::CudaQ4G64V0, 8, 256,
                                   p4->codes, p4->scales);
          auto u8 = unpack_cuda_v0(LogicalQuantizerId::Q8G32V0,
                                   PhysicalLayoutId::CudaQ8G32V0, 8, 256,
                                   p8->codes, p8->scales);
          expect(static_cast<bool>(p4) && static_cast<bool>(u4) &&
                     u4->codes == q4->codes && u4->scales == q4->scales,
                 "real Q4 sample pack/unpack");
          expect(static_cast<bool>(p8) && static_cast<bool>(u8) &&
                     u8->codes == q8->codes,
                 "real Q8 sample pack/unpack");
          auto dq = dequantize_to_bf16(*q4);
          std::vector<std::uint16_t> x(256, 0x3C00);
          auto gemv = qw38::compiler::reference_gemv_packed(*p4, x);
          auto gemv2 = qw38::compiler::reference_gemv_bf16(*dq, x, 8, 256);
          expect(static_cast<bool>(dq) && static_cast<bool>(gemv) &&
                     static_cast<bool>(gemv2) && *gemv == *gemv2,
                 "real Q4 reference contraction");
        }
      }
    }
  } else {
    std::cout << "quant compiler authority evidence skipped (explicit "
                 "extended checkpoint evidence)\n";
  }

  if (g_failures != 0) {
    std::cerr << g_failures << " failures\n";
    return 1;
  }
  return 0;
}
