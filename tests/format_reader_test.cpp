#include "format_reader_support.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>

using qw38::format::Artifact;
using qw38::format::error_message;
using qw38::format::FormatErrorCode;
using qw38::format::IntegrityKind;
using qw38::format::PhysicalLayoutId;
using qw38::format::SharedBinding;
using qw38::format::StorageClass;
using qw38::format::test::mutate_schema;
using qw38::format::test::read_all;
using qw38::format::test::record_boundaries;
using qw38::format::test::ScratchDir;
using qw38::format::test::write_minimal;
using qw38::format::test::write_task003;
using qw38::format::test::write_task003_mutated;

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

void expect_code(std::expected<Artifact, qw38::format::FormatError> const& r,
                 FormatErrorCode code, std::string_view what) {
  if (r) {
    fail(std::string(what) + ": unexpectedly succeeded");
    return;
  }
  if (r.error().code != code) {
    fail(std::string(what) + ": got " + error_message(r.error()));
  }
}

void test_truncation_at_record_boundaries() {
  ScratchDir dir("qw38-reader-trunc");
  auto fx = write_task003(dir.file("fixture.qw38"));
  if (fx.path.empty()) {
    fail("write task003 fixture for truncation");
    return;
  }
  auto bytes = read_all(fx.path);
  auto art = Artifact::open(fx.path);
  if (!art) {
    fail(std::string("open valid fixture: ") + error_message(art.error()));
    return;
  }
  auto bounds = record_boundaries(art->header(), art->schema());
  expect(!bounds.empty(), "record boundaries are nonempty");
  expect(bounds.back() == bytes.size(),
         "walker last boundary is the full artifact size");

  int checked = 0;
  for (auto b : bounds) {
    if (b >= bytes.size()) {
      continue;
    }
    auto truncated = std::vector<std::byte>(bytes.begin(),
                                            bytes.begin() + static_cast<std::ptrdiff_t>(b));
    auto parsed = Artifact::parse(truncated);
    if (parsed) {
      fail("truncation at " + std::to_string(b) + " was accepted");
    }
    ++checked;
  }
  expect(checked >= 20, "checked many record-boundary truncations");
}

void test_open_and_parse_roundtrip_minimal() {
  ScratchDir dir("qw38-reader-min");
  auto fx = write_minimal(dir.file("min.qw38"));
  if (fx.path.empty()) {
    fail("write minimal fixture");
    return;
  }
  auto opened = Artifact::open(fx.path);
  if (!opened) {
    fail(std::string("open minimal: ") + error_message(opened.error()));
    return;
  }
  expect(opened->find_tensor("v") != nullptr, "logical identity v exists");
  expect(opened->find_tensor(1) != nullptr, "tensor id 1 exists");
  expect(opened->find_tensor("missing") == nullptr, "unknown name is null");
  auto payload = opened->payload("v");
  if (!payload) {
    fail(error_message(payload.error()));
    return;
  }
  expect(payload->size() == fx.payload.size(), "payload length 8");
  expect(std::equal(payload->begin(), payload->end(), fx.payload.begin()),
         "payload bytes match");
  auto scales = opened->scales("v");
  expect(scales && scales->empty(), "unquantized tensor has empty scales");
  auto missing = opened->payload("nope");
  if (missing) {
    fail("missing payload lookup succeeded");
  } else {
    expect(missing.error().code == FormatErrorCode::TensorNotFound,
           "unknown logical identity is tensor_not_found");
  }
  expect(opened->compiler().ident == "qw38", "compiler revision");
  expect(opened->state().size() == 3, "minimal fixture carries V0 state schema");
  expect(opened->identity().path == fx.path, "identity path");

  auto bytes = read_all(fx.path);
  auto parsed = Artifact::parse(bytes);
  if (!parsed) {
    fail(std::string("parse minimal: ") + error_message(parsed.error()));
    return;
  }
  auto p2 = parsed->payload("v");
  expect(p2 && std::equal(p2->begin(), p2->end(), fx.payload.begin()),
         "parse views match open");
}

void test_bad_magic_version_enum() {
  ScratchDir dir("qw38-reader-magic");
  auto fx = write_minimal(dir.file("m.qw38"));
  auto bytes = read_all(fx.path);
  bytes[0] = std::byte{'X'};
  expect_code(Artifact::parse(bytes), FormatErrorCode::BadMagic, "bad magic");

  bytes = read_all(fx.path);
  bytes[8] = std::byte{2};
  expect_code(Artifact::parse(bytes), FormatErrorCode::UnsupportedContainerVersion,
              "unsupported container version");

  bytes = read_all(fx.path);
  bytes[10] = std::byte{9};
  expect_code(Artifact::parse(bytes), FormatErrorCode::UnsupportedManifestVersion,
              "unsupported header manifest version");

  auto mutated = mutate_schema(read_all(fx.path), [](auto& schema) {
    schema.manifest_version = 9;
  });
  if (!mutated) {
    fail(error_message(mutated.error()));
    return;
  }
  expect_code(Artifact::parse(*mutated),
              FormatErrorCode::UnsupportedManifestVersion,
              "unsupported schema manifest version");

  bytes = read_all(fx.path);
  bool replaced_layout = false;
  for (std::size_t i = qw38::format::kHeaderSizeV0; i + 1 < bytes.size();
       ++i) {
    if (bytes[i] == std::byte{0x05} && bytes[i + 1] == std::byte{0x02}) {
      bytes[i] = std::byte{0xFF};
      bytes[i + 1] = std::byte{0xFF};
      replaced_layout = true;
      break;
    }
  }
  expect(replaced_layout, "minimal manifest contains its BF16 vector layout");
  expect_code(Artifact::parse(bytes),
              FormatErrorCode::UnsupportedPhysicalLayoutVersion,
              "unknown layout enum");
}

void test_misalignment_overflow_overlap() {
  ScratchDir dir("qw38-reader-arith");
  auto fx = write_minimal(dir.file("a.qw38"));
  auto file = read_all(fx.path);

  auto misaligned = mutate_schema(file, [](auto& schema) {
    schema.tensors[0].payload.offset += 1;
  });
  if (!misaligned) {
    fail(error_message(misaligned.error()));
    return;
  }
  expect_code(Artifact::parse(*misaligned), FormatErrorCode::Misaligned,
              "misaligned payload");

  auto overflowed = file;
  // manifest_offset = UINT64_MAX-10, length = 32 → add overflow in header.
  overflowed[16] = std::byte{0xF6};
  overflowed[17] = std::byte{0xFF};
  overflowed[18] = std::byte{0xFF};
  overflowed[19] = std::byte{0xFF};
  overflowed[20] = std::byte{0xFF};
  overflowed[21] = std::byte{0xFF};
  overflowed[22] = std::byte{0xFF};
  overflowed[23] = std::byte{0xFF};
  overflowed[24] = std::byte{32};
  overflowed[25] = std::byte{0};
  overflowed[26] = std::byte{0};
  overflowed[27] = std::byte{0};
  overflowed[28] = std::byte{0};
  overflowed[29] = std::byte{0};
  overflowed[30] = std::byte{0};
  overflowed[31] = std::byte{0};
  expect_code(Artifact::parse(overflowed), FormatErrorCode::Overflow,
              "header manifest arithmetic overflow");

  auto fx2 = write_task003(dir.file("ov.qw38"));
  auto two = read_all(fx2.path);
  auto overlap = mutate_schema(two, [](auto& schema) {
    schema.tensors[1].payload.offset = schema.tensors[0].payload.offset;
  });
  if (!overlap) {
    fail(error_message(overlap.error()));
    return;
  }
  expect_code(Artifact::parse(*overlap), FormatErrorCode::OverlappingSpan,
              "overlapping unique payloads");
}

void test_noncanonical_directory_and_span_order() {
  ScratchDir dir("qw38-reader-order");
  auto fx = write_task003(dir.file("order.qw38"));
  auto const file = read_all(fx.path);

  auto directory = mutate_schema(file, [](auto& schema) {
    std::swap(schema.tensors[1], schema.tensors[2]);
  });
  if (!directory) {
    fail(error_message(directory.error()));
    return;
  }
  expect_code(Artifact::parse(*directory), FormatErrorCode::InvalidSpan,
              "noncanonical tensor directory order");

  auto spans = mutate_schema(file, [](auto& schema) {
    auto& tensor = schema.tensors[1];
    tensor.payload.offset = 768;
    tensor.scales.offset = 512;
    for (auto& rec : schema.integrity) {
      if (rec.tensor_id != tensor.tensor_id) {
        continue;
      }
      if (rec.kind == IntegrityKind::Sha256PayloadSpan) {
        rec.region = tensor.payload;
      } else if (rec.kind == IntegrityKind::Sha256ScaleSpan) {
        rec.region = tensor.scales;
      }
    }
  });
  if (!spans) {
    fail(error_message(spans.error()));
    return;
  }
  expect_code(Artifact::parse(*spans), FormatErrorCode::InvalidSpan,
              "scale span preceding its payload");
}

void test_invalid_pairs_shapes_scales_shared_state() {
  ScratchDir dir("qw38-reader-schema");
  auto fx = write_task003(dir.file("s.qw38"));
  auto file = read_all(fx.path);

  auto pair = mutate_schema(file, [](auto& schema) {
    schema.tensors[1].layout = PhysicalLayoutId::CudaBf16DenseTileV0;
  });
  if (!pair) {
    fail(error_message(pair.error()));
    return;
  }
  expect_code(Artifact::parse(*pair), FormatErrorCode::InvalidQuantizerLayoutPair,
              "invalid layout/quantizer pair");

  auto storage = mutate_schema(file, [](auto& schema) {
    schema.tensors[1].storage = StorageClass::Bf16;
  });
  if (!storage) {
    fail(error_message(storage.error()));
    return;
  }
  expect_code(Artifact::parse(*storage),
              FormatErrorCode::InvalidStorageQuantizerPair,
              "invalid storage/quantizer pair");

  auto valid = Artifact::parse(file);
  if (!valid) {
    fail(error_message(valid.error()));
    return;
  }
  auto shape = file;
  auto const& schema = valid->schema();
  std::size_t const first_shape =
      static_cast<std::size_t>(valid->header().manifest_offset) + 2 +
      (2 + schema.compiler.ident.size() + 16) + 3 * 32 +
      (4 + schema.precision.bindings.size() * 4) + 2 + 4 + 4 +
      (2 + schema.tensors[0].logical_name.size());
  // Keep the valid rank and corrupt logical[0] directly. Object-level encode
  // now rejects malformed shapes, while the reader still needs hostile bytes.
  for (std::size_t i = 0; i < 8; ++i) {
    shape[first_shape + 8 + i] = std::byte{0};
  }
  expect_code(Artifact::parse(shape), FormatErrorCode::InvalidShape, "bad shape");

  auto scales = mutate_schema(file, [](auto& schema) {
    schema.tensors[1].scales.length += 2;
  });
  if (!scales) {
    fail(error_message(scales.error()));
    return;
  }
  expect_code(Artifact::parse(*scales), FormatErrorCode::InvalidSpan,
              "scale length mismatch");

  auto shared = mutate_schema(file, [](auto& schema) {
    schema.shared_bindings[0].alias_tensor_id =
        schema.shared_bindings[0].owner_tensor_id;
  });
  if (!shared) {
    fail(error_message(shared.error()));
    return;
  }
  expect_code(Artifact::parse(*shared), FormatErrorCode::SharedBinding,
              "self shared binding");

  auto reverse = mutate_schema(file, [](auto& schema) {
    auto const binding = schema.shared_bindings[0];
    schema.shared_bindings.push_back(
        SharedBinding{.owner_tensor_id = binding.alias_tensor_id,
                      .alias_tensor_id = binding.owner_tensor_id});
  });
  if (!reverse) {
    fail(error_message(reverse.error()));
    return;
  }
  expect_code(Artifact::parse(*reverse), FormatErrorCode::SharedBinding,
              "reverse shared-binding pair");

  auto state = mutate_schema(file, [](auto& schema) {
    schema.state[0].layer_count = 7;
  });
  if (!state) {
    fail(error_message(state.error()));
    return;
  }
  expect_code(Artifact::parse(*state), FormatErrorCode::InvalidStateAllocation,
              "bad GDN state schema");

  auto live = mutate_schema(file, [](auto& schema) {
    schema.state[0].live_payload_present = true;
  });
  if (!live) {
    fail(error_message(live.error()));
    return;
  }
  expect_code(Artifact::parse(*live), FormatErrorCode::LiveStateForbidden,
              "live state forbidden");
}

void test_required_state_scratch_and_precision_schema() {
  ScratchDir dir("qw38-reader-runtime-schema");
  auto fx = write_task003(dir.file("runtime-schema.qw38"));
  auto const file = read_all(fx.path);

  auto expect_mutation = [&](auto&& mutator, FormatErrorCode code,
                             std::string_view what) {
    auto mutated = mutate_schema(file, mutator);
    if (!mutated) {
      fail(std::string(what) + ": " + error_message(mutated.error()));
      return;
    }
    expect_code(Artifact::parse(*mutated), code, what);
  };

  expect_mutation(
      [](auto& schema) { schema.state.erase(schema.state.begin()); },
      FormatErrorCode::InvalidStateAllocation, "missing required state");
  expect_mutation(
      [](auto& schema) { schema.state[0].total_bytes += 4; },
      FormatErrorCode::InvalidStateAllocation, "wrong fixed state byte count");
  expect_mutation(
      [](auto& schema) {
        schema.state[1].dtype = qw38::format::ArithmeticDtype::Fp32;
      },
      FormatErrorCode::InvalidStateAllocation, "wrong state dtype");
  expect_mutation(
      [](auto& schema) { schema.state[2].shape_per_layer.padded[1] += 1; },
      FormatErrorCode::InvalidStateAllocation, "padded KV state");
  expect_mutation(
      [](auto& schema) { schema.scratch.pop_back(); },
      FormatErrorCode::InvalidScratchAllocation, "missing required scratch");
  expect_mutation(
      [](auto& schema) { schema.scratch[0].bytes -= 4; },
      FormatErrorCode::InvalidScratchAllocation, "invalid scratch byte count");
  expect_mutation(
      [](auto& schema) {
        schema.scratch[2].dtype = qw38::format::ArithmeticDtype::Fp32;
      },
      FormatErrorCode::InvalidScratchAllocation, "wrong scratch dtype");
  expect_mutation(
      [](auto& schema) {
        for (auto& binding : schema.precision.bindings) {
          if (binding.domain ==
              qw38::format::PrecisionDomain::GdnRecurrentState) {
            binding.dtype = qw38::format::ArithmeticDtype::Bf16;
          }
        }
      },
      FormatErrorCode::InvalidPrecisionPolicy,
      "contradictory persistent-state precision policy");

  for (std::size_t i = 0; i < 3; ++i) {
    expect_mutation(
        [i](auto& schema) { schema.state[i].shape_per_layer.padded[0] += 1; },
        FormatErrorCode::InvalidStateAllocation,
        "locked state representation rejects padded geometry");
  }
  for (std::size_t i = 0; i < 7; ++i) {
    expect_mutation(
        [i](auto& schema) {
          schema.scratch[i].dtype =
              schema.scratch[i].dtype == qw38::format::ArithmeticDtype::Fp32
                  ? qw38::format::ArithmeticDtype::Bf16
                  : qw38::format::ArithmeticDtype::Fp32;
        },
        FormatErrorCode::InvalidScratchAllocation,
        "locked scratch representation rejects contradictory dtype");
    expect_mutation(
        [i](auto& schema) { schema.scratch[i].bytes += 4; },
        FormatErrorCode::InvalidScratchAllocation,
        "locked scratch representation rejects contradictory bytes");
  }
}

void test_canonical_owner_range_validation() {
  ScratchDir dir("qw38-reader-owner");
  auto fx = write_task003(dir.file("owner.qw38"));
  auto file = read_all(fx.path);

  auto header_overlap = mutate_schema(file, [](auto& schema) {
    schema.tensors[0].payload.offset = 0;
    schema.tensors[3].payload.offset = 0;
  });
  if (!header_overlap) {
    fail(error_message(header_overlap.error()));
    return;
  }
  expect_code(Artifact::parse(*header_overlap),
              FormatErrorCode::OverlappingSpan,
              "canonical alias payload cannot overlap the header");
}

void test_bad_hash_and_leftover() {
  ScratchDir dir("qw38-reader-hash");
  auto fx = write_minimal(dir.file("h.qw38"));
  auto bytes = read_all(fx.path);
  bytes[256] = static_cast<std::byte>(static_cast<std::uint8_t>(bytes[256]) ^ 0xFF);
  expect(static_cast<bool>(Artifact::parse(bytes)),
         "payload byte flip does not trigger a payload digest scan");

  bytes = read_all(fx.path);
  auto opened = Artifact::open(fx.path);
  auto const& recs = opened->schema().integrity;
  expect(!recs.empty(), "integrity records present");
  auto const digest_off =
      opened->header().manifest_offset + opened->header().manifest_length - 32;
  bytes[static_cast<std::size_t>(digest_off)] =
      static_cast<std::byte>(static_cast<std::uint8_t>(
                                 bytes[static_cast<std::size_t>(digest_off)]) ^
                             0x01);
  expect_code(Artifact::parse(bytes), FormatErrorCode::IntegrityDigestMismatch,
              "manifest digest flip");

  bytes = read_all(fx.path);
  bytes.push_back(std::byte{0xAB});
  expect_code(Artifact::parse(bytes), FormatErrorCode::LeftoverBytes,
              "trailing leftover bytes");
}

void test_manifest_record_field_corruption() {
  ScratchDir dir("qw38-reader-manifest-record");
  auto fx = write_minimal(dir.file("manifest-record.qw38"));
  auto const pristine = read_all(fx.path);
  auto opened = Artifact::open(fx.path);
  if (!opened) {
    fail(std::string("open manifest record fixture: ") +
         error_message(opened.error()));
    return;
  }

  auto const record_offset = opened->header().manifest_offset +
                             opened->header().manifest_length -
                             qw38::format::kIntegrityRecordBytes;
  struct Field {
    std::string_view name;
    std::uint64_t offset;
    FormatErrorCode expected;
  };
  // IntegrityRecord is kind:u16, reserved:u16, tensor_id:u32,
  // region.offset:u64, region.length:u64, digest:32 bytes.
  std::array<Field, 6> const fields{{
      {"kind", 0, FormatErrorCode::UnknownEnum},
      {"reserved", 2, FormatErrorCode::ReservedNonzero},
      {"tensor id", 4, FormatErrorCode::IntegrityDigestMismatch},
      {"region offset", 8, FormatErrorCode::InvalidSpan},
      {"region length", 16, FormatErrorCode::InvalidSpan},
      {"digest", 24, FormatErrorCode::IntegrityDigestMismatch},
  }};
  for (auto const& field : fields) {
    auto corrupted = pristine;
    auto const byte_offset = record_offset + field.offset;
    corrupted[static_cast<std::size_t>(byte_offset)] =
        static_cast<std::byte>(static_cast<std::uint8_t>(
                                    corrupted[static_cast<std::size_t>(byte_offset)]) ^
                                0x01u);
    expect_code(Artifact::parse(corrupted), field.expected,
                std::string("manifest record ") + std::string(field.name) +
                    " byte corruption");
  }
}

void test_manifest_limit_precedes_span_copy() {
  ScratchDir dir("qw38-reader-limit");
  auto fx = write_minimal(dir.file("limit.qw38"));
  auto bytes = read_all(fx.path);
  auto const hostile_length = qw38::format::kMaxManifestBytesV0 + 1;
  for (std::size_t i = 0; i < 8; ++i) {
    bytes[24 + i] =
        static_cast<std::byte>((hostile_length >> (8 * i)) & 0xFFu);
  }

  auto const borrowed = std::span<std::byte const>{bytes};
  expect_code(Artifact::parse(borrowed),
              FormatErrorCode::ResourceLimitExceeded,
              "oversized manifest is rejected before span ownership copy");
}

void test_quantized_payload_domains() {
  ScratchDir dir("qw38-reader-quantized-domains");
  auto expect_malformed = [&](std::string_view name, auto&& mutate) {
    auto fx = write_task003_mutated(dir.file(std::string(name) + ".qw38"),
                                    std::forward<decltype(mutate)>(mutate));
    if (fx.path.empty()) {
      fail(std::string(name) + ": failed to write fixture");
      return;
    }
    expect_code(Artifact::open(fx.path),
                FormatErrorCode::InvalidQuantizedPayload, name);
  };

  expect_malformed("forbidden Q4 code", [](auto&, auto& fx) {
    fx.q4_payload[0] = std::byte{0x08};
  });
  expect_malformed("forbidden Q8 code", [](auto&, auto& fx) {
    fx.q8_payload[0] = std::byte{0x80};
  });
  expect_malformed("nonzero Q4 code padding", [](auto& schema, auto& fx) {
    schema.tensors[1].shape.logical[0] = 7;
    std::fill(fx.q4_payload.begin() + 7 * 128, fx.q4_payload.end(),
              std::byte{0});
    std::fill(fx.q4_scales.begin() + 7 * 8, fx.q4_scales.end(),
              std::byte{0});
    fx.q4_payload[7 * 128] = std::byte{0x01};
  });
  expect_malformed("nonzero Q4 scale padding", [](auto& schema, auto& fx) {
    schema.tensors[1].shape.logical[0] = 7;
    std::fill(fx.q4_payload.begin() + 7 * 128, fx.q4_payload.end(),
              std::byte{0});
  });

  struct BadScale {
    std::uint16_t bits;
    char const* name;
  };
  for (auto const bad : {BadScale{0x0001u, "subnormal FP16 scale"},
                         BadScale{0x8000u, "negative zero FP16 scale"},
                         BadScale{0xBC00u, "negative FP16 scale"},
                         BadScale{0x7C00u, "infinite FP16 scale"},
                         BadScale{0x7E00u, "NaN FP16 scale"}}) {
    expect_malformed(bad.name, [bad](auto&, auto& fx) {
      fx.q4_scales[0] = static_cast<std::byte>(bad.bits & 0xFFu);
      fx.q4_scales[1] =
          static_cast<std::byte>((bad.bits >> 8) & 0xFFu);
    });
  }

  auto direct = write_task003_mutated(
      dir.file("direct-unpack.qw38"), [](auto&, auto& fx) {
        fx.q4_payload[0] = std::byte{0x08};
      });
  auto unpacked = qw38::format::unpack_cuda_v0(
      qw38::format::LogicalQuantizerId::Q4G64V0,
      qw38::format::PhysicalLayoutId::CudaQ4G64V0, 8, 256,
      direct.q4_payload, direct.q4_scales);
  expect(!unpacked &&
             unpacked.error().code ==
                 FormatErrorCode::InvalidQuantizedPayload,
         "independent unpacker rejects the artifact domain rejected before CUDA");
}

void test_q4k_artifact() {
  using namespace qw38::format;
  ScratchDir dir("qw38-reader-q4k");
  auto schema = test::base_schema();
  schema.precision = candidate_v2_precision_policy();
  TensorRecord t;
  t.tensor_id = 1;
  t.logical_name = "q4k";
  t.shape = test::rank2(8, 256);
  t.storage = StorageClass::Int4Grouped;
  t.quantizer = LogicalQuantizerId::Q4KCandidateV2;
  t.layout = PhysicalLayoutId::CudaQ4KCandidateV2;
  t.mapping = {.kind = MappingKind::DenseTileNK, .tile_rows = 8,
               .tile_k = 256, .group_size = 256, .packed_bytes_per_tile_row = 128};
  schema.tensors.push_back(t);
  auto writer = ArtifactWriter::create(dir.file("q4k.qw38"), schema);
  expect(writer.has_value(), "Q4_K writer accepts distinct format and policy");
  if (!writer) return;
  std::vector<std::byte> codes(8 * 128, std::byte{0xf8});
  std::vector<std::byte> metadata(8 * 16);
  for (std::size_t row = 0; row < 8; ++row) {
    metadata[row * 16] = std::byte{1};
    metadata[row * 16 + 2] = std::byte{2};
    for (std::size_t i = 4; i < 16; ++i) metadata[row * 16 + i] = std::byte{0xff};
  }
  expect(writer->write_span("q4k", SpanKind::Payload, codes).has_value(), "Q4_K write codes");
  expect(writer->write_span("q4k", SpanKind::Scales, metadata).has_value(), "Q4_K write metadata");
  expect(writer->finalize().has_value(), "Q4_K finalize");
  auto artifact = Artifact::open(dir.file("q4k.qw38"));
  expect(artifact.has_value(), "Q4_K reader accepts unsigned 8/15 and subnormal scales");
  if (!artifact) return;
  auto const& saved = artifact->schema();
  expect(saved.precision.id == PrecisionPolicyId::CandidateV2 &&
             saved.tensors[0].scales.length == 128 &&
             saved.tensors[0].payload.length == 1024,
         "Q4_K serialized IDs and byte extents survive roundtrip");
  for (auto const policy : {PrecisionPolicyId::V0, PrecisionPolicyId::CandidateV1}) {
    auto invalid = saved;
    invalid.precision.id = policy;
    auto status = validate_schema(invalid);
    expect(!status && status.error().code == FormatErrorCode::InvalidPrecisionPolicy,
           "Q4_K cannot use an older precision policy");
  }
  auto invalid = saved;
  invalid.tensors[0].mapping.group_size = 64;
  expect(!validate_schema(invalid), "Q4_K schema rejects Q4G64 grouping");
  invalid = saved;
  invalid.tensors[0].scales.length = 64;
  expect(!validate_schema(invalid), "Q4_K schema rejects Q4G64 metadata extent");
  invalid = saved;
  invalid.tensors[0].layout = PhysicalLayoutId::CudaQ4G64CandidateV1;
  expect(!validate_schema(invalid), "Q4_K schema rejects old physical identity");
  auto bytes = read_all(dir.file("q4k.qw38"));
  bytes[saved.tensors[0].scales.offset + 1] = std::byte{0x7c};
  expect_code(Artifact::parse(bytes), FormatErrorCode::InvalidQuantizedPayload,
              "Q4_K reader rejects NaN metadata before CUDA");
}

}  // namespace

int main() {
  test_truncation_at_record_boundaries();
  test_open_and_parse_roundtrip_minimal();
  test_bad_magic_version_enum();
  test_misalignment_overflow_overlap();
  test_noncanonical_directory_and_span_order();
  test_invalid_pairs_shapes_scales_shared_state();
  test_required_state_scratch_and_precision_schema();
  test_canonical_owner_range_validation();
  test_bad_hash_and_leftover();
  test_manifest_record_field_corruption();
  test_manifest_limit_precedes_span_copy();
  test_quantized_payload_domains();
  test_q4k_artifact();
  if (g_failures != 0) {
    std::cerr << g_failures << " reader unit checks failed\n";
    return 1;
  }
  std::cout << "format reader unit ok\n";
  return 0;
}
