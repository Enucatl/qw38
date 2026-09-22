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
  auto const schema = qw38::format::v0_language_scratch_schema();
  return {schema.begin(), schema.end()};
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
