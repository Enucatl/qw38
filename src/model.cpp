#include "model.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <utility>

namespace qw38::internal {
namespace {

constexpr std::uint64_t kMaxMetadata = 100000;
constexpr std::uint64_t kMaxTensors = 100000;
constexpr std::uint64_t kMaxString = 64ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t kMaxArray = 10000000;
constexpr std::uint32_t kMaxDimensions = 8;

class Reader final {
 public:
  explicit Reader(const std::string& path) noexcept
      : input_(path, std::ios::binary | std::ios::ate) {
    if (input_) {
      const std::streampos end = input_.tellg();
      if (end >= 0) {
        size_ = static_cast<std::uint64_t>(end);
        input_.seekg(0);
      }
    }
  }

  bool open() const noexcept { return input_.is_open(); }
  std::uint64_t size() const noexcept { return size_; }
  std::uint64_t position() noexcept {
    const std::streampos position = input_.tellg();
    return position < 0 ? size_ + 1 : static_cast<std::uint64_t>(position);
  }

  bool bytes(void* output, std::uint64_t count) noexcept {
    if (count > size_ || position() > size_ - count ||
        count > static_cast<std::uint64_t>(
                    std::numeric_limits<std::streamsize>::max())) {
      return false;
    }
    input_.read(static_cast<char*>(output), static_cast<std::streamsize>(count));
    return input_.good();
  }

  bool skip(std::uint64_t count) noexcept {
    if (count > size_ || position() > size_ - count ||
        count > static_cast<std::uint64_t>(
                    std::numeric_limits<std::streamoff>::max())) {
      return false;
    }
    input_.seekg(static_cast<std::streamoff>(count), std::ios::cur);
    return input_.good();
  }

  bool u32(std::uint32_t* output) noexcept {
    std::array<unsigned char, 4> data{};
    if (!bytes(data.data(), data.size())) {
      return false;
    }
    *output = static_cast<std::uint32_t>(data[0]) |
              (static_cast<std::uint32_t>(data[1]) << 8U) |
              (static_cast<std::uint32_t>(data[2]) << 16U) |
              (static_cast<std::uint32_t>(data[3]) << 24U);
    return true;
  }

  bool u64(std::uint64_t* output) noexcept {
    std::array<unsigned char, 8> data{};
    if (!bytes(data.data(), data.size())) {
      return false;
    }
    *output = 0;
    for (std::uint32_t index = 0; index < 8; ++index) {
      *output |= static_cast<std::uint64_t>(data[index]) << (index * 8U);
    }
    return true;
  }

  bool string(std::string* output) noexcept {
    std::uint64_t length = 0;
    if (!u64(&length) || length > kMaxString || length > size_ ||
        position() > size_ - length) {
      return false;
    }
    output->resize(static_cast<std::size_t>(length));
    return length == 0 || bytes(output->data(), length);
  }

 private:
  std::ifstream input_;
  std::uint64_t size_ = 0;
};

std::uint64_t scalar_bytes(std::uint32_t type) noexcept {
  switch (type) {
    case 0:
    case 1:
    case 7:
      return 1;
    case 2:
    case 3:
      return 2;
    case 4:
    case 5:
    case 6:
      return 4;
    case 10:
    case 11:
    case 12:
      return 8;
    default:
      return 0;
  }
}

bool skip_value(Reader* reader, std::uint32_t type, std::uint32_t depth) noexcept {
  if (depth > 4) {
    return false;
  }
  if (type == 8) {
    std::string ignored;
    return reader->string(&ignored);
  }
  if (type == 9) {
    std::uint32_t element_type = 0;
    std::uint64_t count = 0;
    if (!reader->u32(&element_type) || !reader->u64(&count) || count > kMaxArray) {
      return false;
    }
    const std::uint64_t width = scalar_bytes(element_type);
    if (width != 0) {
      return count <= std::numeric_limits<std::uint64_t>::max() / width &&
             reader->skip(count * width);
    }
    for (std::uint64_t index = 0; index < count; ++index) {
      if (!skip_value(reader, element_type, depth + 1)) {
        return false;
      }
    }
    return true;
  }
  const std::uint64_t width = scalar_bytes(type);
  return width != 0 && reader->skip(width);
}

bool read_unsigned_value(Reader* reader, std::uint32_t type,
                         std::uint64_t* output) noexcept {
  if (type == 4) {
    std::uint32_t value = 0;
    if (!reader->u32(&value)) {
      return false;
    }
    *output = value;
    return true;
  }
  if (type == 10) {
    return reader->u64(output);
  }
  return false;
}

Status malformed(const std::string& detail) noexcept {
  return {StatusCode::kIncompatibleArtifact, "malformed GGUF: " + detail};
}

}  // namespace

Status inspect_gguf(const std::string& path, ModelInfo* info) noexcept {
  if (info == nullptr || path.empty()) {
    return {StatusCode::kInvalidArgument, "GGUF path and output are required"};
  }
  Reader reader(path);
  if (!reader.open()) {
    return {StatusCode::kIoError, "cannot open GGUF"};
  }
  std::array<char, 4> magic{};
  std::uint64_t tensor_count = 0;
  ModelInfo parsed;
  if (!reader.bytes(magic.data(), magic.size()) || magic != std::array<char, 4>{'G', 'G', 'U', 'F'} ||
      !reader.u32(&parsed.gguf_version) || !reader.u64(&tensor_count) ||
      !reader.u64(&parsed.metadata_count)) {
    return malformed("truncated or invalid header");
  }
  if (parsed.gguf_version != 3) {
    return malformed("only GGUF v3 is accepted");
  }
  if (tensor_count == 0 || tensor_count > kMaxTensors ||
      parsed.metadata_count == 0 || parsed.metadata_count > kMaxMetadata) {
    return malformed("implausible tensor or metadata count");
  }

  for (std::uint64_t index = 0; index < parsed.metadata_count; ++index) {
    std::string key;
    std::uint32_t type = 0;
    if (!reader.string(&key) || !reader.u32(&type)) {
      return malformed("truncated metadata key");
    }
    if (key == "general.architecture" || key == "general.name") {
      if (type != 8) {
        return malformed("string metadata has wrong type");
      }
      std::string* target = key == "general.architecture" ? &parsed.architecture : &parsed.name;
      if (!reader.string(target)) {
        return malformed("truncated string metadata");
      }
      continue;
    }
    std::uint64_t* target = nullptr;
    if (key == "general.alignment") target = &parsed.alignment;
    if (key == "qwen35.block_count") target = &parsed.block_count;
    if (key == "qwen35.context_length") target = &parsed.context_length;
    if (key == "qwen35.embedding_length") target = &parsed.embedding_length;
    if (key == "qwen35.attention.head_count") target = &parsed.query_heads;
    if (key == "qwen35.attention.head_count_kv") target = &parsed.kv_heads;
    if (key == "qwen35.rope.dimension_count") target = &parsed.rope_dimensions;
    if (target != nullptr) {
      if (!read_unsigned_value(&reader, type, target)) {
        return malformed("numeric metadata has wrong type");
      }
    } else if (!skip_value(&reader, type, 0)) {
      return malformed("unsupported or truncated metadata value");
    }
  }

  parsed.tensors.reserve(static_cast<std::size_t>(tensor_count));
  for (std::uint64_t index = 0; index < tensor_count; ++index) {
    TensorInfo tensor;
    std::uint32_t dimensions = 0;
    if (!reader.string(&tensor.name) || tensor.name.empty() ||
        !reader.u32(&dimensions) || dimensions == 0 ||
        dimensions > kMaxDimensions) {
      return malformed("invalid tensor descriptor");
    }
    tensor.dimensions.resize(dimensions);
    for (std::uint64_t& dimension : tensor.dimensions) {
      if (!reader.u64(&dimension) || dimension == 0) {
        return malformed("invalid tensor dimension");
      }
    }
    if (!reader.u32(&tensor.type) || !reader.u64(&tensor.offset)) {
      return malformed("truncated tensor descriptor");
    }
    parsed.tensors.push_back(std::move(tensor));
  }

  if (parsed.alignment == 0 || (parsed.alignment & (parsed.alignment - 1)) != 0 ||
      parsed.alignment > 4096) {
    return malformed("invalid alignment");
  }
  const std::uint64_t descriptor_end = reader.position();
  const std::uint64_t remainder = descriptor_end % parsed.alignment;
  parsed.data_offset = descriptor_end + (remainder == 0 ? 0 : parsed.alignment - remainder);
  if (parsed.data_offset > reader.size()) {
    return malformed("tensor data begins past end of file");
  }

  std::vector<std::size_t> order(parsed.tensors.size());
  for (std::size_t index = 0; index < order.size(); ++index) order[index] = index;
  std::sort(order.begin(), order.end(), [&parsed](std::size_t left, std::size_t right) {
    return parsed.tensors[left].offset < parsed.tensors[right].offset;
  });
  for (std::size_t index = 0; index < order.size(); ++index) {
    TensorInfo& tensor = parsed.tensors[order[index]];
    const std::uint64_t absolute = parsed.data_offset + tensor.offset;
    const std::uint64_t next = index + 1 == order.size()
                                   ? reader.size()
                                   : parsed.data_offset + parsed.tensors[order[index + 1]].offset;
    if (absolute < parsed.data_offset || absolute > reader.size() || next <= absolute ||
        next > reader.size() || tensor.offset % parsed.alignment != 0) {
      return malformed("overlapping, unaligned, or out-of-range tensor data");
    }
    tensor.bytes = next - absolute;
  }
  *info = std::move(parsed);
  return Status::ok();
}

Status validate_qwen38_contract(const ModelInfo& info) noexcept {
  if (info.architecture != "qwen35" || info.name != "Qwen3.8-27B" ||
      info.block_count != 64 || info.context_length != 262144 ||
      info.embedding_length != 5120 || info.query_heads != 24 ||
      info.kv_heads != 4 || info.rope_dimensions != 64 ||
      info.tensors.size() != 851) {
    return {StatusCode::kIncompatibleArtifact,
            "GGUF does not match the pinned Qwen3.8-27B contract"};
  }
  return Status::ok();
}

}  // namespace qw38::internal
