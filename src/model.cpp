#include "model.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <set>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

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

bool read_string_array(Reader* reader, std::uint32_t type,
                       std::vector<std::string>* output) noexcept {
  std::uint32_t element_type = 0;
  std::uint64_t count = 0;
  if (type != 9 || !reader->u32(&element_type) || element_type != 8 ||
      !reader->u64(&count) || count > kMaxArray) {
    return false;
  }
  output->reserve(static_cast<std::size_t>(count));
  for (std::uint64_t index = 0; index < count; ++index) {
    std::string value;
    if (!reader->string(&value)) return false;
    output->push_back(std::move(value));
  }
  return true;
}

bool read_u32_array(Reader* reader, std::uint32_t type,
                    std::vector<std::uint32_t>* output) noexcept {
  std::uint32_t element_type = 0;
  std::uint64_t count = 0;
  if (type != 9 || !reader->u32(&element_type) ||
      (element_type != 4 && element_type != 5) || !reader->u64(&count) ||
      count > kMaxArray) {
    return false;
  }
  output->resize(static_cast<std::size_t>(count));
  for (std::uint32_t& value : *output) {
    if (!reader->u32(&value)) return false;
  }
  return true;
}

Status malformed(const std::string& detail) noexcept {
  return {StatusCode::kIncompatibleArtifact, "malformed GGUF: " + detail};
}

bool tensor_storage_bytes(const TensorInfo& tensor,
                          std::uint64_t* bytes) noexcept {
  std::uint64_t elements = 1;
  for (std::uint64_t dimension : tensor.dimensions) {
    if (dimension > std::numeric_limits<std::uint64_t>::max() / elements) {
      return false;
    }
    elements *= dimension;
  }
  std::uint64_t block_elements = 0;
  std::uint64_t block_bytes = 0;
  switch (tensor.type) {
    case 0:  // F32
      block_elements = 1;
      block_bytes = 4;
      break;
    case 8:  // Q8_0
      block_elements = 32;
      block_bytes = 34;
      break;
    case 12:  // Q4_K
      block_elements = 256;
      block_bytes = 144;
      break;
    case 13:  // Q5_K
      block_elements = 256;
      block_bytes = 176;
      break;
    case 14:  // Q6_K
      block_elements = 256;
      block_bytes = 210;
      break;
    default:
      return false;
  }
  if (elements % block_elements != 0 ||
      elements / block_elements >
          std::numeric_limits<std::uint64_t>::max() / block_bytes) {
    return false;
  }
  *bytes = elements / block_elements * block_bytes;
  return true;
}

bool dimensions_equal(const TensorInfo& tensor,
                      std::initializer_list<std::uint64_t> expected) noexcept {
  return tensor.dimensions.size() == expected.size() &&
         std::equal(tensor.dimensions.begin(), tensor.dimensions.end(),
                    expected.begin());
}

struct ExpectedTensor final {
  std::string name;
  std::initializer_list<std::uint64_t> dimensions;
  std::uint32_t type;
  const char* role;
};

Status admit_tensor(std::vector<TensorInfo>* tensors,
                    std::set<std::string>* admitted,
                    const ExpectedTensor& expected) noexcept {
  const auto match = std::find_if(
      tensors->begin(), tensors->end(), [&expected](const TensorInfo& tensor) {
        return tensor.name == expected.name;
      });
  if (match == tensors->end() || match->type != expected.type ||
      !dimensions_equal(*match, expected.dimensions)) {
    return {StatusCode::kIncompatibleArtifact,
            "tensor contract mismatch: " + expected.name};
  }
  match->semantic_role = expected.role;
  admitted->insert(expected.name);
  return Status::ok();
}

bool is_quant_type(std::uint32_t type) noexcept {
  return type == 8 || type == 12 || type == 13 || type == 14;
}

Status admit_named(std::vector<TensorInfo>* tensors,
                   std::set<std::string>* admitted, const std::string& name,
                   const std::vector<std::uint64_t>& dimensions,
                   bool quant_matrix, const char* role) noexcept {
  const auto match = std::find_if(
      tensors->begin(), tensors->end(),
      [&name](const TensorInfo& tensor) { return tensor.name == name; });
  if (match == tensors->end() || match->dimensions != dimensions) {
    return {StatusCode::kIncompatibleArtifact,
            "tensor contract mismatch: " + name};
  }
  if (quant_matrix) {
    if (!is_quant_type(match->type)) {
      return {StatusCode::kIncompatibleArtifact,
              "tensor contract mismatch: " + name};
    }
  } else if (match->type != 0) {
    return {StatusCode::kIncompatibleArtifact,
            "tensor contract mismatch: " + name};
  }
  match->semantic_role = role;
  admitted->insert(name);
  return Status::ok();
}

}  // namespace

MappedFile::MappedFile() noexcept = default;
MappedFile::~MappedFile() { close(); }

MappedFile::MappedFile(MappedFile&& other) noexcept
    : descriptor_(other.descriptor_), data_(other.data_), size_(other.size_) {
  other.descriptor_ = -1;
  other.data_ = nullptr;
  other.size_ = 0;
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
  if (this != &other) {
    close();
    descriptor_ = other.descriptor_;
    data_ = other.data_;
    size_ = other.size_;
    other.descriptor_ = -1;
    other.data_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

Status MappedFile::open(const std::string& path) noexcept {
  close();
  descriptor_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (descriptor_ < 0) {
    return {StatusCode::kIoError, "cannot open model for mapping"};
  }
  struct stat attributes {};
  if (fstat(descriptor_, &attributes) != 0 || attributes.st_size <= 0 ||
      static_cast<std::uintmax_t>(attributes.st_size) >
          std::numeric_limits<std::size_t>::max()) {
    close();
    return {StatusCode::kIoError, "cannot size model mapping"};
  }
  size_ = static_cast<std::size_t>(attributes.st_size);
  void* mapping = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, descriptor_, 0);
  if (mapping == MAP_FAILED) {
    data_ = nullptr;
    close();
    return {StatusCode::kResourceExhausted, "cannot mmap model"};
  }
  data_ = static_cast<unsigned char*>(mapping);
  return Status::ok();
}

const unsigned char* MappedFile::data() const noexcept { return data_; }
std::size_t MappedFile::size() const noexcept { return size_; }

void MappedFile::close() noexcept {
  if (data_ != nullptr) munmap(data_, size_);
  if (descriptor_ >= 0) ::close(descriptor_);
  descriptor_ = -1;
  data_ = nullptr;
  size_ = 0;
}

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
    if (key == "general.architecture" || key == "general.name" ||
        key == "tokenizer.ggml.model") {
      if (type != 8) {
        return malformed("string metadata has wrong type");
      }
      std::string* target = &parsed.tokenizer_model;
      if (key == "general.architecture") target = &parsed.architecture;
      if (key == "general.name") target = &parsed.name;
      if (!reader.string(target)) {
        return malformed("truncated string metadata");
      }
      continue;
    }
    if (key == "tokenizer.ggml.tokens" || key == "tokenizer.ggml.merges") {
      std::vector<std::string>* target = key == "tokenizer.ggml.tokens"
                                             ? &parsed.tokenizer_tokens
                                             : &parsed.tokenizer_merges;
      if (!read_string_array(&reader, type, target)) {
        return malformed("tokenizer string array has wrong type or is truncated");
      }
      continue;
    }
    if (key == "tokenizer.ggml.token_type") {
      if (!read_u32_array(&reader, type, &parsed.tokenizer_token_types)) {
        return malformed("tokenizer type array has wrong type or is truncated");
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
    if (tensor.offset > reader.size() - parsed.data_offset) {
      return malformed("tensor offset overflows or exceeds file");
    }
    const std::uint64_t absolute = parsed.data_offset + tensor.offset;
    std::uint64_t next = reader.size();
    if (index + 1 != order.size()) {
      const std::uint64_t next_offset = parsed.tensors[order[index + 1]].offset;
      if (next_offset > reader.size() - parsed.data_offset) {
        return malformed("tensor offset overflows or exceeds file");
      }
      next = parsed.data_offset + next_offset;
    }
    if (absolute < parsed.data_offset || absolute > reader.size() || next <= absolute ||
        next > reader.size() || tensor.offset % parsed.alignment != 0) {
      return malformed("overlapping, unaligned, or out-of-range tensor data");
    }
    tensor.padded_span_bytes = next - absolute;
    if (!tensor_storage_bytes(tensor, &tensor.storage_bytes) ||
        tensor.storage_bytes > tensor.padded_span_bytes) {
      return malformed("unsupported type or invalid quantized tensor size");
    }
  }
  *info = std::move(parsed);
  return Status::ok();
}

Status validate_qwen38_contract(ModelInfo* info) noexcept {
  if (info == nullptr) {
    return {StatusCode::kInvalidArgument, "model contract output is required"};
  }
  if (info->architecture != "qwen35" || info->name != "Qwen3.8-27B" ||
      info->block_count != 64 || info->context_length != 262144 ||
      info->embedding_length != 5120 || info->query_heads != 24 ||
      info->kv_heads != 4 || info->rope_dimensions != 64 ||
      info->tensors.size() != 851 || info->tokenizer_model != "gpt2" ||
      info->tokenizer_tokens.size() != 248320 ||
      info->tokenizer_token_types.size() != info->tokenizer_tokens.size() ||
      info->tokenizer_merges.size() != 247587) {
    return {StatusCode::kIncompatibleArtifact,
            "GGUF does not match the pinned Qwen3.8-27B contract"};
  }
  std::set<std::string> admitted;
  Status status = admit_tensor(
      &info->tensors, &admitted,
      {"output.weight", {5120, 248320}, 14, "output_projection"});
  if (!status.is_ok()) return status;
  status = admit_tensor(&info->tensors, &admitted,
                        {"output_norm.weight", {5120}, 0, "final_norm"});
  if (!status.is_ok()) return status;
  status = admit_tensor(&info->tensors, &admitted,
                        {"token_embd.weight", {5120, 248320}, 12,
                         "token_embedding"});
  if (!status.is_ok()) return status;

  for (std::uint32_t layer = 0; layer < 64; ++layer) {
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    const bool attention = layer % 4 == 3;
    const std::array<ExpectedTensor, 5> common = {{
        {prefix + "attn_norm.weight", {5120}, 0, "input_norm"},
        {prefix + "ffn_down.weight", {17408, 5120}, 12, "ffn_down"},
        {prefix + "ffn_gate.weight", {5120, 17408}, 12, "ffn_gate"},
        {prefix + "ffn_up.weight", {5120, 17408}, 12, "ffn_up"},
        {prefix + "post_attention_norm.weight", {5120}, 0, "ffn_norm"},
    }};
    for (const ExpectedTensor& expected : common) {
      status = admit_tensor(&info->tensors, &admitted, expected);
      if (!status.is_ok()) return status;
    }
    if (attention) {
      const std::array<ExpectedTensor, 6> expected = {{
          {prefix + "attn_k.weight", {5120, 1024}, 8, "attention_k"},
          {prefix + "attn_k_norm.weight", {256}, 0, "attention_k_norm"},
          {prefix + "attn_output.weight", {6144, 5120}, 14,
           "attention_output"},
          {prefix + "attn_q.weight", {5120, 12288}, 8, "attention_q_gate"},
          {prefix + "attn_q_norm.weight", {256}, 0, "attention_q_norm"},
          {prefix + "attn_v.weight", {5120, 1024}, 8, "attention_v"},
      }};
      for (const ExpectedTensor& tensor : expected) {
        status = admit_tensor(&info->tensors, &admitted, tensor);
        if (!status.is_ok()) return status;
      }
    } else {
      const std::array<ExpectedTensor, 9> expected = {{
          {prefix + "attn_gate.weight", {5120, 6144}, 8, "gdn_value_gate"},
          {prefix + "attn_qkv.weight", {5120, 10240}, 8, "gdn_packed_qkv"},
          {prefix + "ssm_a", {48}, 0, "gdn_decay"},
          {prefix + "ssm_alpha.weight", {5120, 48}, 8, "gdn_alpha"},
          {prefix + "ssm_beta.weight", {5120, 48}, 8, "gdn_beta"},
          {prefix + "ssm_conv1d.weight", {4, 10240}, 0, "gdn_convolution"},
          {prefix + "ssm_dt.bias", {48}, 0, "gdn_dt_bias"},
          {prefix + "ssm_norm.weight", {128}, 0, "gdn_norm"},
          {prefix + "ssm_out.weight", {6144, 5120}, 8, "gdn_output"},
      }};
      for (const ExpectedTensor& tensor : expected) {
        status = admit_tensor(&info->tensors, &admitted, tensor);
        if (!status.is_ok()) return status;
      }
    }
  }
  if (admitted.size() != info->tensors.size()) {
    return {StatusCode::kIncompatibleArtifact,
            "GGUF contains unrecognized or duplicate tensors"};
  }
  return Status::ok();
}

Status validate_qwen35_2b_contract(ModelInfo* info) noexcept {
  if (info == nullptr) {
    return {StatusCode::kInvalidArgument, "model contract output is required"};
  }
  const ModelGeometry geometry = qwen35_2b_geometry();
  const bool has_output = std::any_of(
      info->tensors.begin(), info->tensors.end(), [](const TensorInfo& tensor) {
        return tensor.name == "output.weight";
      });
  const std::size_t expected_tensors =
      geometry.expected_tensor_count + (has_output ? 1 : 0);
  if (info->architecture != "qwen35" || info->block_count != 24 ||
      info->embedding_length != 2048 || info->query_heads != 8 ||
      info->kv_heads != 2 || info->rope_dimensions != 64 ||
      info->tensors.size() != expected_tensors ||
      info->tokenizer_model != "gpt2" ||
      info->tokenizer_tokens.size() != 248320 ||
      info->tokenizer_token_types.size() != info->tokenizer_tokens.size() ||
      info->tokenizer_merges.size() != 247587) {
    return {StatusCode::kIncompatibleArtifact,
            "GGUF does not match the admitted Qwen3.5-2B contract"};
  }
  std::set<std::string> admitted;
  Status status = admit_named(&info->tensors, &admitted, "token_embd.weight",
                              {2048, 248320}, true, "token_embedding");
  if (status.is_ok()) {
    status = admit_named(&info->tensors, &admitted, "output_norm.weight", {2048},
                         false, "final_norm");
  }
  if (status.is_ok() && has_output) {
    status = admit_named(&info->tensors, &admitted, "output.weight",
                         {2048, 248320}, true, "output_projection");
  }
  for (std::uint32_t layer = 0; status.is_ok() && layer < 24; ++layer) {
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    const bool attention = layer % 4 == 3;
    status = admit_named(&info->tensors, &admitted, prefix + "attn_norm.weight",
                         {2048}, false, "input_norm");
    if (status.is_ok()) {
      status = admit_named(&info->tensors, &admitted, prefix + "ffn_down.weight",
                           {6144, 2048}, true, "ffn_down");
    }
    if (status.is_ok()) {
      status = admit_named(&info->tensors, &admitted, prefix + "ffn_gate.weight",
                           {2048, 6144}, true, "ffn_gate");
    }
    if (status.is_ok()) {
      status = admit_named(&info->tensors, &admitted, prefix + "ffn_up.weight",
                           {2048, 6144}, true, "ffn_up");
    }
    if (status.is_ok()) {
      status = admit_named(&info->tensors, &admitted,
                           prefix + "post_attention_norm.weight", {2048}, false,
                           "ffn_norm");
    }
    if (!status.is_ok()) break;
    if (attention) {
      status = admit_named(&info->tensors, &admitted, prefix + "attn_k.weight",
                           {2048, 512}, true, "attention_k");
      if (status.is_ok()) {
        status = admit_named(&info->tensors, &admitted,
                             prefix + "attn_k_norm.weight", {256}, false,
                             "attention_k_norm");
      }
      if (status.is_ok()) {
        status = admit_named(&info->tensors, &admitted,
                             prefix + "attn_output.weight", {2048, 2048}, true,
                             "attention_output");
      }
      if (status.is_ok()) {
        status = admit_named(&info->tensors, &admitted, prefix + "attn_q.weight",
                             {2048, 4096}, true, "attention_q_gate");
      }
      if (status.is_ok()) {
        status = admit_named(&info->tensors, &admitted,
                             prefix + "attn_q_norm.weight", {256}, false,
                             "attention_q_norm");
      }
      if (status.is_ok()) {
        status = admit_named(&info->tensors, &admitted, prefix + "attn_v.weight",
                             {2048, 512}, true, "attention_v");
      }
    } else {
      status = admit_named(&info->tensors, &admitted, prefix + "attn_gate.weight",
                           {2048, 2048}, true, "gdn_value_gate");
      if (status.is_ok()) {
        status = admit_named(&info->tensors, &admitted,
                             prefix + "attn_qkv.weight", {2048, 6144}, true,
                             "gdn_packed_qkv");
      }
      if (status.is_ok()) {
        status = admit_named(&info->tensors, &admitted, prefix + "ssm_a", {16},
                             false, "gdn_decay");
      }
      if (status.is_ok()) {
        status = admit_named(&info->tensors, &admitted,
                             prefix + "ssm_alpha.weight", {2048, 16}, true,
                             "gdn_alpha");
      }
      if (status.is_ok()) {
        status = admit_named(&info->tensors, &admitted,
                             prefix + "ssm_beta.weight", {2048, 16}, true,
                             "gdn_beta");
      }
      if (status.is_ok()) {
        status = admit_named(&info->tensors, &admitted,
                             prefix + "ssm_conv1d.weight", {4, 6144}, false,
                             "gdn_convolution");
      }
      if (status.is_ok()) {
        status = admit_named(&info->tensors, &admitted, prefix + "ssm_dt.bias",
                             {16}, false, "gdn_dt_bias");
      }
      if (status.is_ok()) {
        status = admit_named(&info->tensors, &admitted,
                             prefix + "ssm_norm.weight", {128}, false,
                             "gdn_norm");
      }
      if (status.is_ok()) {
        status = admit_named(&info->tensors, &admitted, prefix + "ssm_out.weight",
                             {2048, 2048}, true, "gdn_output");
      }
    }
  }
  if (!status.is_ok()) return status;
  if (admitted.size() != info->tensors.size()) {
    return {StatusCode::kIncompatibleArtifact,
            "GGUF contains unrecognized or duplicate tensors"};
  }
  return Status::ok();
}

Status admit_pinned_geometry(ModelInfo* info, ModelGeometry* geometry) noexcept {
  if (info == nullptr || geometry == nullptr) {
    return {StatusCode::kInvalidArgument,
            "model info and geometry output are required"};
  }
  Status status = validate_qwen38_contract(info);
  if (status.is_ok()) {
    *geometry = qwen38_27b_geometry();
    return status;
  }
  status = validate_qwen35_2b_contract(info);
  if (!status.is_ok()) return status;
  *geometry = qwen35_2b_geometry();
  if (info->tensors.size() == geometry->expected_tensor_count + 1) {
    geometry->expected_tensor_count += 1;
    geometry->tied_embeddings = false;
  }
  return Status::ok();
}

const char* ggml_type_name(std::uint32_t type) noexcept {
  switch (type) {
    case 0:
      return "F32";
    case 8:
      return "Q8_0";
    case 12:
      return "Q4_K";
    case 13:
      return "Q5_K";
    case 14:
      return "Q6_K";
    default:
      return "UNSUPPORTED";
  }
}

}  // namespace qw38::internal
