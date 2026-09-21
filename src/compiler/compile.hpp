#pragma once

#include "compiler/checkpoint.hpp"
#include "compiler/error.hpp"
#include "compiler/identity.hpp"
#include "compiler/quantization/reference.hpp"
#include "format/writer.hpp"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <vector>

namespace qw38::compiler {

struct CompileOptions {
  qw38::format::CompilerRevision revision{
      .ident = kCompilerIdent,
      .major = kCompilerMajor,
      .minor = kCompilerMinor,
      .patch = kCompilerPatch};
  bool verify_reconstruction{true};
  WeightFormatPolicy format_policy{WeightFormatPolicy::IdentityBf16};
};

struct CompileResult {
  qw38::format::ArtifactIdentity identity{};
  std::uint64_t peak_rss_bytes{};
  std::uint32_t language_instances{};
  std::uint32_t mtp_instances{};
  std::uint32_t included_tensors{};
  std::uint32_t vision_excluded{};
};

[[nodiscard]] std::vector<qw38::format::StateAllocation> language_state_schema();
[[nodiscard]] std::vector<qw38::format::ScratchAllocation> language_scratch_schema();

[[nodiscard]] std::expected<qw38::format::ArtifactSchema, CompilerError>
build_schema(ClassifiedCheckpoint const& classified,
             qw38::format::Hash256 const& source_hash,
             qw38::format::Hash256 const& config_hash,
             qw38::format::Hash256 const& tokenizer_hash,
             qw38::format::CompilerRevision const& revision,
             WeightFormatPolicy policy);

[[nodiscard]] std::expected<qw38::format::ArtifactSchema, CompilerError>
build_identity_schema(ClassifiedCheckpoint const& classified,
                      qw38::format::Hash256 const& source_hash,
                      qw38::format::Hash256 const& config_hash,
                      qw38::format::Hash256 const& tokenizer_hash,
                      qw38::format::CompilerRevision const& revision);

[[nodiscard]] std::expected<CompileResult, CompilerError> compile_checkpoint(
    std::filesystem::path const& checkpoint,
    std::filesystem::path const& output, CompileOptions const& options = {});

[[nodiscard]] std::expected<CompileResult, CompilerError> compile_identity(
    std::filesystem::path const& checkpoint,
    std::filesystem::path const& output, CompileOptions const& options = {});

[[nodiscard]] std::expected<void, CompilerError> verify_identity_artifact(
    std::filesystem::path const& artifact,
    std::filesystem::path const& checkpoint);

[[nodiscard]] std::expected<void, CompilerError> verify_compiled_artifact(
    std::filesystem::path const& artifact,
    std::filesystem::path const& checkpoint, WeightFormatPolicy policy);

struct SyntheticTensor {
  ExpectedTensor expected;
  std::vector<std::byte> bytes;
};

[[nodiscard]] std::expected<CompileResult, CompilerError> compile_synthetic(
    std::filesystem::path const& output,
    qw38::format::Hash256 const& source_hash,
    qw38::format::Hash256 const& config_hash,
    qw38::format::Hash256 const& tokenizer_hash,
    qw38::format::CompilerRevision const& revision,
    std::vector<SyntheticTensor> tensors,
    WeightFormatPolicy policy = WeightFormatPolicy::IdentityBf16);

[[nodiscard]] std::uint32_t count_instances(
    std::span<qw38::format::GraphBinding const> bindings,
    SourceClass which) noexcept;

[[nodiscard]] std::uint64_t current_peak_rss_bytes();

}  // namespace qw38::compiler
