#pragma once

#include "format/error.hpp"
#include "format/schema.hpp"

#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string_view>
#include <sys/types.h>

namespace qw38::format {

enum class SpanKind : std::uint8_t {
  Payload = 1,
  Scales = 2,
};

struct ArtifactIdentity {
  std::filesystem::path path;
  Hash256 manifest_digest{};
  std::uint64_t size_bytes{};
  std::uint64_t manifest_offset{};
  std::uint64_t manifest_length{};
  CompilerRevision compiler{};

  friend bool operator==(ArtifactIdentity const&,
                         ArtifactIdentity const&) = default;
};

// Explicit syscall boundary for deterministic writer I/O tests. Production
// callers use the default POSIX operations by leaving this unset.
struct WriterFilesystem {
  ssize_t (*pwrite)(void* context, int fd, void const* data, std::size_t size,
                    off_t offset){};
  int (*fsync)(void* context, int fd){};
  int (*close)(void* context, int fd){};
  int (*rename)(void* context, char const* from, char const* to){};
  int (*remove)(void* context, char const* path){};
  void* context{};
};

// Streaming host writer. Payload and scale bytes are hashed and stored as they
// arrive; the immutable input schema is copied and its span/integrity slots are
// filled by the writer. Failed finalization never publishes a destination file.
class ArtifactWriter {
 public:
  ArtifactWriter(ArtifactWriter&&) noexcept;
  ArtifactWriter& operator=(ArtifactWriter&&) noexcept;
  ~ArtifactWriter();

  ArtifactWriter(ArtifactWriter const&) = delete;
  ArtifactWriter& operator=(ArtifactWriter const&) = delete;

  [[nodiscard]] static std::expected<ArtifactWriter, FormatError> create(
      std::filesystem::path const& destination, ArtifactSchema const& schema,
      WriterFilesystem const* filesystem = nullptr);

  // Append a nonempty chunk of a named span. Repeated calls with the same
  // (logical_name, kind) stream into one span until the layout length is met.
  // Alias names write the owner's unique physical span.
  [[nodiscard]] std::expected<void, FormatError> write_span(
      std::string_view logical_name, SpanKind kind,
      std::span<std::byte const> chunk);

  [[nodiscard]] std::expected<ArtifactIdentity, FormatError> finalize();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  explicit ArtifactWriter(std::unique_ptr<Impl> impl) noexcept;
};

class ArtifactBuilder {
 public:
  ArtifactBuilder& destination(std::filesystem::path path);
  ArtifactBuilder& schema(ArtifactSchema schema);

  [[nodiscard]] std::expected<ArtifactWriter, FormatError> open() const;

 private:
  std::filesystem::path destination_;
  ArtifactSchema schema_{};
  bool has_destination_{false};
  bool has_schema_{false};
};

}  // namespace qw38::format
