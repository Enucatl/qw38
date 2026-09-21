#pragma once

#include "format/format.hpp"
#include "format_reader_support.hpp"
#include "runtime/sizes.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace qw38::runtime::test {

inline std::vector<qw38::format::ScratchAllocation> language_scratch() {
  using qw38::format::ArithmeticDtype;
  using qw38::format::ScratchAllocation;
  using qw38::format::ScratchKind;
  return {
      ScratchAllocation{.kind = ScratchKind::ResidualH,
                        .dtype = ArithmeticDtype::Fp32,
                        .bytes = kResidualBytesPerToken},
      ScratchAllocation{.kind = ScratchKind::ResidualHMid,
                        .dtype = ArithmeticDtype::Fp32,
                        .bytes = kResidualBytesPerToken},
      ScratchAllocation{.kind = ScratchKind::NormalizedHidden,
                        .dtype = ArithmeticDtype::Bf16,
                        .bytes = kNormalizedBytesPerToken},
      ScratchAllocation{.kind = ScratchKind::GdnWorkspace,
                        .dtype = ArithmeticDtype::Fp32,
                        .bytes = kGdnWorkspaceBytesPerToken},
      ScratchAllocation{.kind = ScratchKind::AttentionWorkspace,
                        .dtype = ArithmeticDtype::Fp32,
                        .bytes = kAttentionWorkspaceBytesPerToken},
      ScratchAllocation{.kind = ScratchKind::MlpSwiglu,
                        .dtype = ArithmeticDtype::Bf16,
                        .bytes = kSwigluBytesPerToken},
      ScratchAllocation{.kind = ScratchKind::Logits,
                        .dtype = ArithmeticDtype::Fp32,
                        .bytes = kLogitsBytesPerToken},
  };
}

struct RuntimeFixture {
  std::filesystem::path path;
  std::array<std::byte, 8> payload{
      std::byte{0x00}, std::byte{0x11}, std::byte{0x22}, std::byte{0x33},
      std::byte{0x44}, std::byte{0x55}, std::byte{0x66}, std::byte{0x77}};
};

inline RuntimeFixture write_language_fixture(std::filesystem::path dest) {
  using qw38::format::ArtifactWriter;
  using qw38::format::SpanKind;
  RuntimeFixture fx;
  fx.path = std::move(dest);
  auto schema = qw38::format::test::base_schema();
  schema.tensors.push_back(
      qw38::format::test::unplaced_bf16_vector(1, "v", 4));
  auto state = language_persistent_schema();
  schema.state.assign(state.begin(), state.end());
  schema.scratch = language_scratch();
  auto writer = ArtifactWriter::create(fx.path, schema);
  if (!writer) {
    fx.path.clear();
    return fx;
  }
  if (!writer->write_span("v", SpanKind::Payload, fx.payload) ||
      !writer->finalize()) {
    fx.path.clear();
    return fx;
  }
  return fx;
}

}  // namespace qw38::runtime::test
