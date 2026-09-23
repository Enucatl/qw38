#include "compiler/checkpoint.hpp"

#include "compiler/json.hpp"
#include "format/sha256.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <unistd.h>

#include <sys/mman.h>
#include <sys/stat.h>

namespace qw38::compiler {
namespace {

using qw38::format::Hash256;
using qw38::format::Sha256;

CompilerError io(std::string_view field, std::string_view detail) {
  return make_error(CompilerErrorCode::IoFailure, field, detail);
}

std::expected<std::uint64_t, CompilerError> require_u64(Json const& obj,
                                                        std::string_view key,
                                                        std::string_view field) {
  auto const* v = obj.find(key);
  if (v == nullptr || !v->is_number()) {
    return std::unexpected(make_error(CompilerErrorCode::InvalidConfig, field,
                                      std::string(key) + " missing"));
  }
  auto n = parse_json_u64(*v, std::string(field) + "." + std::string(key));
  if (!n) {
    return std::unexpected(n.error());
  }
  return *n;
}

std::expected<bool, CompilerError> require_bool(Json const& obj,
                                                std::string_view key,
                                                std::string_view field) {
  auto const* v = obj.find(key);
  if (v == nullptr || !v->is_bool()) {
    return std::unexpected(make_error(CompilerErrorCode::InvalidConfig, field,
                                      std::string(key) + " missing"));
  }
  return v->as_bool();
}

std::expected<double, CompilerError> require_number(Json const& obj,
                                                    std::string_view key,
                                                    std::string_view field) {
  auto const* v = obj.find(key);
  if (v == nullptr || !v->is_number() || !std::isfinite(v->as_number())) {
    return std::unexpected(make_error(CompilerErrorCode::InvalidConfig, field,
                                      std::string(key) +
                                          " missing or nonfinite"));
  }
  return v->as_number();
}

std::expected<std::uint32_t, CompilerError> require_u32(
    Json const& obj, std::string_view key, std::string_view field) {
  auto value = require_u64(obj, key, field);
  if (!value) {
    return std::unexpected(value.error());
  }
  if (*value > std::numeric_limits<std::uint32_t>::max()) {
    return std::unexpected(make_error(CompilerErrorCode::InvalidConfig, field,
                                      std::string(key) + " exceeds uint32"));
  }
  return static_cast<std::uint32_t>(*value);
}

std::expected<std::string, CompilerError> require_string(
    Json const& obj, std::string_view key, std::string_view field) {
  auto const* v = obj.find(key);
  if (v == nullptr || !v->is_string()) {
    return std::unexpected(make_error(CompilerErrorCode::InvalidConfig, field,
                                      std::string(key) + " missing"));
  }
  return v->as_string();
}

std::expected<Json const*, CompilerError> require_object(
    Json const& obj, std::string_view key, std::string_view field) {
  auto const* v = obj.find(key);
  if (v == nullptr || !v->is_object()) {
    return std::unexpected(make_error(CompilerErrorCode::InvalidConfig, field,
                                      std::string(key) + " missing object"));
  }
  return v;
}

std::expected<std::uint64_t, CompilerError> read_u64_le(
    std::span<std::byte const> bytes, std::uint64_t offset,
    std::string_view field) {
  if (offset > bytes.size() || bytes.size() - offset < 8) {
    return std::unexpected(io(field, "truncated u64"));
  }
  std::uint64_t v = 0;
  for (int i = 0; i < 8; ++i) {
    v |= static_cast<std::uint64_t>(
             std::to_integer<std::uint8_t>(bytes[offset + static_cast<std::uint64_t>(i)]))
         << (8 * i);
  }
  return v;
}

}  // namespace

std::expected<std::uint64_t, CompilerError> parse_json_u64(
    Json const& value, std::string_view field) {
  if (!value.is_number()) {
    return std::unexpected(make_error(CompilerErrorCode::InvalidJson, field,
                                      "expected an integer"));
  }
  double const number = value.as_number();
  if (!std::isfinite(number)) {
    return std::unexpected(make_error(CompilerErrorCode::InvalidJson, field,
                                      "integer must be finite"));
  }
  if (number < 0) {
    return std::unexpected(make_error(CompilerErrorCode::InvalidJson, field,
                                      "integer must be nonnegative"));
  }
  if (std::trunc(number) != number) {
    return std::unexpected(make_error(CompilerErrorCode::InvalidJson, field,
                                      "integer must not be fractional"));
  }
  // 2^64 is exactly representable as double, while UINT64_MAX is not.
  if (number >= std::ldexp(1.0, 64)) {
    return std::unexpected(make_error(CompilerErrorCode::InvalidJson, field,
                                      "integer is not representable as uint64"));
  }
  return static_cast<std::uint64_t>(number);
}

std::expected<std::vector<std::uint64_t>, CompilerError>
parse_safetensors_shape(Json const& value, std::string_view field) {
  if (!value.is_array()) {
    return std::unexpected(make_error(CompilerErrorCode::InvalidJson, field,
                                      "shape is not an array"));
  }
  auto const& values = value.as_array();
  if (values.empty() || values.size() > qw38::format::kMaxRank) {
    return std::unexpected(make_error(CompilerErrorCode::ShapeMismatch, field,
                                      "shape rank must be 1..8"));
  }
  std::vector<std::uint64_t> shape;
  shape.reserve(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    auto dim = parse_json_u64(values[i],
                              std::string(field) + "[" + std::to_string(i) + "]");
    if (!dim) {
      return std::unexpected(dim.error());
    }
    shape.push_back(*dim);
  }
  return shape;
}

MappedShard::MappedShard(MappedShard&& other) noexcept
    : fd_(other.fd_), addr_(other.addr_), size_(other.size_) {
  other.fd_ = -1;
  other.addr_ = MAP_FAILED;
  other.size_ = 0;
}

MappedShard& MappedShard::operator=(MappedShard&& other) noexcept {
  if (this != &other) {
    this->~MappedShard();
    fd_ = other.fd_;
    addr_ = other.addr_;
    size_ = other.size_;
    other.fd_ = -1;
    other.addr_ = MAP_FAILED;
    other.size_ = 0;
  }
  return *this;
}

MappedShard::~MappedShard() {
  if (addr_ != nullptr && addr_ != MAP_FAILED) {
    ::munmap(addr_, size_);
    addr_ = MAP_FAILED;
  }
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

std::expected<MappedShard, CompilerError> MappedShard::open(
    std::filesystem::path const& path) {
  int raw = -1;
  do {
    raw = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  } while (raw < 0 && errno == EINTR);
  if (raw < 0) {
    return std::unexpected(io(path.string(), std::strerror(errno)));
  }
  struct stat st {};
  if (::fstat(raw, &st) != 0) {
    int e = errno;
    ::close(raw);
    return std::unexpected(io(path.string(), std::strerror(e)));
  }
  if (!S_ISREG(st.st_mode) || st.st_size < 8) {
    ::close(raw);
    return std::unexpected(io(path.string(), "shard is not a safetensors file"));
  }
  auto const size = static_cast<std::size_t>(st.st_size);
  void* addr = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, raw, 0);
  if (addr == MAP_FAILED) {
    int e = errno;
    ::close(raw);
    return std::unexpected(io(path.string(), std::strerror(e)));
  }
  MappedShard mapped;
  mapped.fd_ = raw;
  mapped.addr_ = addr;
  mapped.size_ = size;
  return mapped;
}

std::span<std::byte const> MappedShard::bytes() const noexcept {
  if (addr_ == nullptr || addr_ == MAP_FAILED) {
    return {};
  }
  return {static_cast<std::byte const*>(addr_), size_};
}

std::expected<std::span<std::byte const>, CompilerError>
MappedShard::tensor_bytes(SourceTensor const& tensor) const {
  auto const file = bytes();
  auto const header_len = read_u64_le(file, 0, tensor.name);
  if (!header_len) {
    return std::unexpected(header_len.error());
  }
  if (*header_len > file.size() - 8) {
    return std::unexpected(io(tensor.name, "safetensor header overruns file"));
  }
  auto const payload_base = 8 + *header_len;
  if (tensor.data_offset > file.size() - payload_base) {
    return std::unexpected(make_error(CompilerErrorCode::ShapeMismatch, tensor.name,
                                      "safetensor payload is out of range"));
  }
  auto const start = payload_base + tensor.data_offset;
  if (tensor.nbytes > file.size() - start) {
    return std::unexpected(make_error(CompilerErrorCode::ShapeMismatch, tensor.name,
                                      "safetensor payload is out of range"));
  }
  return file.subspan(static_cast<std::size_t>(start),
                      static_cast<std::size_t>(tensor.nbytes));
}

std::expected<std::string, CompilerError> read_text_file(
    std::filesystem::path const& path, std::string_view field) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return std::unexpected(io(field, "failed to open " + path.string()));
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  if (!in && !in.eof()) {
    return std::unexpected(io(field, "failed to read " + path.string()));
  }
  return ss.str();
}

std::expected<Hash256, CompilerError> hash_file(std::filesystem::path const& path,
                                                std::string_view field) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return std::unexpected(io(field, "failed to open " + path.string()));
  }
  Sha256 hasher;
  std::array<std::byte, 1 << 16> buf{};
  while (in) {
    in.read(reinterpret_cast<char*>(buf.data()),
            static_cast<std::streamsize>(buf.size()));
    auto const n = static_cast<std::size_t>(in.gcount());
    if (n > 0) {
      hasher.update(std::span<std::byte const>{buf.data(), n});
    }
  }
  if (in.bad()) {
    return std::unexpected(io(field, "failed to hash " + path.string()));
  }
  return hasher.finish();
}

std::expected<CheckpointIdentities, CompilerError>
compute_checkpoint_identities(std::filesystem::path const& root) {
  auto const index_path = root / "model.safetensors.index.json";
  auto index_text = read_text_file(index_path, "model.safetensors.index.json");
  if (!index_text) {
    return std::unexpected(index_text.error());
  }
  auto index = parse_json(*index_text, "model.safetensors.index.json");
  if (!index) {
    return std::unexpected(index.error());
  }
  auto const* weight_map = index->find("weight_map");
  if (weight_map == nullptr || !weight_map->is_object()) {
    return std::unexpected(make_error(CompilerErrorCode::InvalidConfig,
                                      "weight_map", "index has no weight_map"));
  }
  for (auto const& [name, shard] : weight_map->as_object()) {
    if (!shard.is_string()) {
      return std::unexpected(make_error(CompilerErrorCode::InvalidConfig, name,
                                        "weight_map value is not a string"));
    }
  }
  CheckpointIdentities identities{};
  // Source identity is metadata-only. The index declares tensor names and
  // shard membership; never hash the containing shard files or tensor payloads.
  auto source = hash_file(index_path, "model.safetensors.index.json");
  if (!source) {
    return std::unexpected(source.error());
  }
  identities.source_hash = *source;
  auto config = hash_file(root / "config.json", "config.json");
  if (!config) {
    return std::unexpected(config.error());
  }
  identities.config_hash = *config;
  auto tokenizer_path = root / "tokenizer.json";
  if (!std::filesystem::exists(tokenizer_path)) {
    tokenizer_path = root / "tokenizer_config.json";
  }
  auto tokenizer = hash_file(tokenizer_path, "tokenizer");
  if (!tokenizer) {
    return std::unexpected(tokenizer.error());
  }
  identities.tokenizer_hash = *tokenizer;
  return identities;
}

std::expected<ArchitectureConfig, CompilerError> parse_text_config(
    std::string_view json_text) {
  auto parsed = parse_json(json_text, "config.json");
  if (!parsed) {
    return std::unexpected(parsed.error());
  }
  Json const& root = *parsed;
  if (!root.is_object()) {
    return std::unexpected(make_error(CompilerErrorCode::InvalidConfig,
                                      "config.json", "root is not an object"));
  }
  ArchitectureConfig cfg{};
  auto arch = root.find("architectures");
  if (arch == nullptr || !arch->is_array() || arch->as_array().size() != 1 ||
      !arch->as_array()[0].is_string()) {
    return std::unexpected(make_error(CompilerErrorCode::InvalidConfig,
                                      "architectures",
                                      "expected a single architecture string"));
  }
  cfg.architecture = arch->as_array()[0].as_string();
  auto model_type = require_string(root, "model_type", "config.json");
  if (!model_type) {
    return std::unexpected(model_type.error());
  }
  cfg.model_type = *model_type;
  auto tie = require_bool(root, "tie_word_embeddings", "config.json");
  if (!tie) {
    return std::unexpected(tie.error());
  }
  cfg.tie_word_embeddings = *tie;

  auto text = require_object(root, "text_config", "config.json");
  if (!text) {
    return std::unexpected(text.error());
  }
  Json const& tc = **text;
  auto text_model_type = require_string(tc, "model_type", "text_config");
  auto hidden = require_u64(tc, "hidden_size", "text_config");
  auto inter = require_u64(tc, "intermediate_size", "text_config");
  auto vocab = require_u64(tc, "vocab_size", "text_config");
  auto max_positions =
      require_u64(tc, "max_position_embeddings", "text_config");
  auto layers = require_u32(tc, "num_hidden_layers", "text_config");
  auto heads = require_u32(tc, "num_attention_heads", "text_config");
  auto kv = require_u32(tc, "num_key_value_heads", "text_config");
  auto head_dim = require_u64(tc, "head_dim", "text_config");
  auto interval = require_u32(tc, "full_attention_interval", "text_config");
  auto conv = require_u64(tc, "linear_conv_kernel_dim", "text_config");
  auto lk = require_u64(tc, "linear_key_head_dim", "text_config");
  auto lv = require_u64(tc, "linear_value_head_dim", "text_config");
  auto nlk = require_u64(tc, "linear_num_key_heads", "text_config");
  auto nlv = require_u64(tc, "linear_num_value_heads", "text_config");
  auto mtp_layers = require_u32(tc, "mtp_num_hidden_layers", "text_config");
  auto dtype = require_string(tc, "dtype", "text_config");
  auto ssm = require_string(tc, "mamba_ssm_dtype", "text_config");
  auto hidden_act = require_string(tc, "hidden_act", "text_config");
  auto output_gate_type =
      require_string(tc, "output_gate_type", "text_config");
  auto mtp_ded = require_bool(tc, "mtp_use_dedicated_embeddings", "text_config");
  auto text_tie = require_bool(tc, "tie_word_embeddings", "text_config");
  auto attention_bias = require_bool(tc, "attention_bias", "text_config");
  auto attn_output_gate = require_bool(tc, "attn_output_gate", "text_config");
  auto use_cache = require_bool(tc, "use_cache", "text_config");
  auto attention_dropout =
      require_number(tc, "attention_dropout", "text_config");
  auto rms_norm_eps = require_number(tc, "rms_norm_eps", "text_config");
  auto prf = require_number(tc, "partial_rotary_factor", "text_config");
  auto rope = require_object(tc, "rope_parameters", "text_config");
  auto layer_types = tc.find("layer_types");
  if (!text_model_type || !hidden || !inter || !vocab || !max_positions ||
      !layers || !heads || !kv || !head_dim || !interval || !conv || !lk ||
      !lv || !nlk || !nlv || !mtp_layers || !dtype || !ssm || !hidden_act ||
      !output_gate_type || !mtp_ded || !text_tie || !attention_bias ||
      !attn_output_gate || !use_cache || !attention_dropout || !rms_norm_eps ||
      !prf || !rope || layer_types == nullptr || !layer_types->is_array()) {
    return std::unexpected(make_error(CompilerErrorCode::InvalidConfig,
                                      "text_config",
                                      "sitting text_config fields are incomplete"));
  }
  cfg.text_model_type = *text_model_type;
  cfg.hidden_size = *hidden;
  cfg.intermediate_size = *inter;
  cfg.vocab_size = *vocab;
  cfg.max_position_embeddings = *max_positions;
  cfg.num_hidden_layers = *layers;
  cfg.num_attention_heads = *heads;
  cfg.num_key_value_heads = *kv;
  cfg.head_dim = *head_dim;
  cfg.full_attention_interval = *interval;
  cfg.linear_conv_kernel_dim = *conv;
  cfg.linear_key_head_dim = *lk;
  cfg.linear_value_head_dim = *lv;
  cfg.linear_num_key_heads = *nlk;
  cfg.linear_num_value_heads = *nlv;
  cfg.mtp_num_hidden_layers = *mtp_layers;
  cfg.dtype = *dtype;
  cfg.mamba_ssm_dtype = *ssm;
  cfg.hidden_act = *hidden_act;
  cfg.output_gate_type = *output_gate_type;
  cfg.mtp_use_dedicated_embeddings = *mtp_ded;
  cfg.text_tie_word_embeddings = *text_tie;
  cfg.attention_bias = *attention_bias;
  cfg.attn_output_gate = *attn_output_gate;
  cfg.use_cache = *use_cache;
  cfg.attention_dropout = *attention_dropout;
  cfg.rms_norm_eps = *rms_norm_eps;
  cfg.partial_rotary_factor = *prf;
  Json const& rp = **rope;
  auto theta = require_number(rp, "rope_theta", "rope_parameters");
  auto rope_partial =
      require_number(rp, "partial_rotary_factor", "rope_parameters");
  auto rope_type = require_string(rp, "rope_type", "rope_parameters");
  auto interleaved = require_bool(rp, "mrope_interleaved", "rope_parameters");
  auto section = rp.find("mrope_section");
  if (!theta || !rope_partial || !rope_type || !interleaved ||
      section == nullptr || !section->is_array()) {
    return std::unexpected(make_error(CompilerErrorCode::InvalidConfig,
                                      "rope_parameters", "incomplete RoPE block"));
  }
  cfg.rope_theta = *theta;
  cfg.rope_partial_rotary_factor = *rope_partial;
  cfg.rope_type = *rope_type;
  cfg.mrope_interleaved = *interleaved;
  for (auto const& item : section->as_array()) {
    auto value = parse_json_u64(item, "mrope_section");
    if (!value || *value > std::numeric_limits<std::uint32_t>::max()) {
      return std::unexpected(
          value ? make_error(CompilerErrorCode::InvalidConfig, "mrope_section",
                             "integer exceeds uint32")
                : value.error());
    }
    cfg.mrope_section.push_back(static_cast<std::uint32_t>(*value));
  }
  for (auto const& item : layer_types->as_array()) {
    if (!item.is_string()) {
      return std::unexpected(make_error(CompilerErrorCode::InvalidConfig,
                                        "layer_types", "expected strings"));
    }
    cfg.layer_types.push_back(item.as_string());
  }
  return cfg;
}

std::expected<void, CompilerError> validate_architecture(
    ArchitectureConfig const& cfg) {
  auto fail = [](std::string_view field, std::string_view detail) {
    return std::unexpected(make_error(CompilerErrorCode::ArchitectureMismatch,
                                      field, detail));
  };
  if (cfg.architecture != kArchitecture) {
    return fail("architectures", "unsupported architecture");
  }
  if (cfg.model_type != kModelType) {
    return fail("model_type", "unsupported multimodal model type");
  }
  if (cfg.text_model_type != kTextModelType) {
    return fail("text_config.model_type", "unsupported language model type");
  }
  if (cfg.tie_word_embeddings != cfg.text_tie_word_embeddings) {
    return fail("tie_word_embeddings",
                "root and text embedding-sharing settings disagree");
  }
  if (cfg.tie_word_embeddings) {
    return fail("tie_word_embeddings", "embeddings and lm_head must remain untied");
  }
  if (cfg.mtp_use_dedicated_embeddings) {
    return fail("mtp_use_dedicated_embeddings",
                "MTP must alias the language embedding payload");
  }
  if (cfg.hidden_size != kHidden) {
    return fail("hidden_size", "unsupported hidden width");
  }
  if (cfg.intermediate_size != kIntermediate) {
    return fail("intermediate_size", "unsupported MLP width");
  }
  if (cfg.vocab_size != kVocab) {
    return fail("vocab_size", "unsupported vocabulary size");
  }
  if (cfg.max_position_embeddings != kMaxPositionEmbeddings) {
    return fail("max_position_embeddings", "unsupported context limit");
  }
  if (cfg.num_hidden_layers != kLayers) {
    return fail("num_hidden_layers", "unsupported language layer count");
  }
  if (cfg.num_attention_heads != kQueryHeads) {
    return fail("num_attention_heads", "unsupported query-head count");
  }
  if (cfg.num_key_value_heads != kKvHeads) {
    return fail("num_key_value_heads", "unsupported KV-head count");
  }
  if (cfg.num_key_value_heads == 0 ||
      cfg.num_attention_heads % cfg.num_key_value_heads != 0) {
    return fail("num_key_value_heads", "query heads must divide into KV groups");
  }
  if (cfg.head_dim != kHeadDim) {
    return fail("head_dim", "unsupported attention head width");
  }
  if (cfg.full_attention_interval != kFullInterval) {
    return fail("full_attention_interval", "unsupported hybrid layer interval");
  }
  if (cfg.linear_conv_kernel_dim != kConvKernel) {
    return fail("linear_conv_kernel_dim", "unsupported causal convolution width");
  }
  if (cfg.linear_key_head_dim != kLinearHeadDim) {
    return fail("linear_key_head_dim", "unsupported GDN key width");
  }
  if (cfg.linear_value_head_dim != kLinearHeadDim) {
    return fail("linear_value_head_dim", "unsupported GDN value width");
  }
  if (cfg.linear_num_key_heads != kLinearKeyHeads) {
    return fail("linear_num_key_heads", "unsupported GDN key-head count");
  }
  if (cfg.linear_num_value_heads != kLinearValueHeads) {
    return fail("linear_num_value_heads", "unsupported GDN value-head count");
  }
  if (cfg.linear_num_key_heads == 0 ||
      cfg.linear_num_value_heads % cfg.linear_num_key_heads != 0) {
    return fail("linear_num_value_heads",
                "GDN value heads must be a multiple of key heads");
  }
  if (cfg.mtp_num_hidden_layers != 1) {
    return fail("mtp_num_hidden_layers", "V0 retains exactly one MTP layer");
  }
  if (cfg.dtype != kWeightDtype) {
    return fail("dtype", "V0 source weights must be bfloat16");
  }
  if (cfg.mamba_ssm_dtype != kMambaSsmDtype) {
    return fail("mamba_ssm_dtype", "V0 GDN state semantics require float32");
  }
  if (cfg.rms_norm_eps != kRmsNormEps) {
    return fail("rms_norm_eps", "unsupported RMS normalization epsilon");
  }
  if (cfg.attention_bias) {
    return fail("attention_bias", "V0 projections are bias-free");
  }
  if (cfg.attention_dropout != kAttentionDropout) {
    return fail("attention_dropout", "V0 attention must be dropout-free");
  }
  if (!cfg.attn_output_gate) {
    return fail("attn_output_gate", "full attention requires its sigmoid gate");
  }
  if (cfg.hidden_act != kHiddenAct) {
    return fail("hidden_act", "V0 MLP activation must be SiLU");
  }
  if (cfg.output_gate_type != kOutputGateType) {
    return fail("output_gate_type", "V0 GDN output gate must be swish");
  }
  if (!cfg.use_cache) {
    return fail("use_cache", "V0 requires persistent decode caches");
  }
  if (cfg.partial_rotary_factor != cfg.rope_partial_rotary_factor) {
    return fail("partial_rotary_factor",
                "text and RoPE partial rotary factors disagree");
  }
  if (cfg.partial_rotary_factor != kPartialRotaryFactor) {
    return fail("partial_rotary_factor", "unsupported partial rotary width");
  }
  if (cfg.rope_theta != kRopeTheta) {
    return fail("rope_theta", "unsupported RoPE base");
  }
  if (cfg.rope_type != kRopeType) {
    return fail("rope_type", "unsupported RoPE variant");
  }
  if (!cfg.mrope_interleaved) {
    return fail("mrope_interleaved", "V0 requires interleaved multimodal RoPE");
  }
  if (cfg.mrope_section != std::vector<std::uint32_t>{11, 11, 10}) {
    return fail("mrope_section", "unsupported multimodal RoPE partition");
  }
  auto const rotary_width =
      static_cast<std::uint64_t>(cfg.head_dim * cfg.partial_rotary_factor);
  std::uint64_t section_sum = 0;
  for (auto const width : cfg.mrope_section) {
    section_sum += width;
  }
  if (rotary_width != kRotaryDim || rotary_width % 2 != 0 ||
      section_sum != rotary_width / 2) {
    return fail("mrope_section",
                "RoPE width and multimodal partition are inconsistent");
  }
  if (cfg.layer_types.size() != kLayers) {
    return fail("layer_types", "expected 64 entries");
  }
  for (std::uint32_t i = 0; i < kLayers; ++i) {
    bool const want_full = is_full_attention_layer(i);
    auto const& got = cfg.layer_types[i];
    if (want_full && got != "full_attention") {
      return fail("layer_types", "full-attention slot mismatch");
    }
    if (!want_full && got != "linear_attention") {
      return fail("layer_types", "linear-attention slot mismatch");
    }
  }
  return {};
}

std::expected<std::vector<SourceTensor>, CompilerError> parse_shard_header(
    std::span<std::byte const> file, std::string const& shard_name) {
  auto header_len = read_u64_le(file, 0, shard_name);
  if (!header_len) {
    return std::unexpected(header_len.error());
  }
  if (*header_len > file.size() - 8) {
    return std::unexpected(io(shard_name, "safetensor header overruns file"));
  }
  auto const json_bytes = file.subspan(8, static_cast<std::size_t>(*header_len));
  std::string json_text(reinterpret_cast<char const*>(json_bytes.data()),
                        json_bytes.size());
  auto parsed = parse_json(json_text, shard_name);
  if (!parsed) {
    return std::unexpected(parsed.error());
  }
  if (!parsed->is_object()) {
    return std::unexpected(make_error(CompilerErrorCode::InvalidJson, shard_name,
                                      "safetensor header is not an object"));
  }
  std::vector<SourceTensor> out;
  auto const payload_base = 8 + *header_len;
  for (auto const& [name, rec] : parsed->as_object()) {
    if (name == "__metadata__") {
      continue;
    }
    if (!rec.is_object()) {
      return std::unexpected(make_error(CompilerErrorCode::InvalidJson, name,
                                        "tensor record is not an object"));
    }
    auto const* dtype = rec.find("dtype");
    auto const* shape = rec.find("shape");
    auto const* offsets = rec.find("data_offsets");
    if (dtype == nullptr || !dtype->is_string() || shape == nullptr ||
        !shape->is_array() || offsets == nullptr || !offsets->is_array() ||
        offsets->as_array().size() != 2) {
      return std::unexpected(make_error(CompilerErrorCode::InvalidJson, name,
                                        "tensor record is incomplete"));
    }
    SourceTensor t{};
    t.name = name;
    t.dtype = dtype->as_string();
    t.shard = shard_name;
    auto parsed_shape = parse_safetensors_shape(*shape, name + ".shape");
    if (!parsed_shape) {
      return std::unexpected(parsed_shape.error());
    }
    t.shape = std::move(*parsed_shape);
    auto start = parse_json_u64(offsets->as_array()[0], name + ".data_offsets[0]");
    auto stop = parse_json_u64(offsets->as_array()[1], name + ".data_offsets[1]");
    if (!start || !stop) {
      return std::unexpected(!start ? start.error() : stop.error());
    }
    if (*stop < *start) {
      return std::unexpected(make_error(CompilerErrorCode::ShapeMismatch, name,
                                        "data_offsets reverse"));
    }
    t.data_offset = *start;
    t.nbytes = *stop - *start;
    if (*stop > file.size() || payload_base > file.size() - *stop) {
      return std::unexpected(make_error(CompilerErrorCode::ShapeMismatch, name,
                                        "payload overruns shard"));
    }
    out.push_back(std::move(t));
  }
  return out;
}

std::expected<Checkpoint, CompilerError> open_checkpoint(
    std::filesystem::path const& root) {
  Checkpoint ckpt;
  ckpt.root = root;
  auto config_text = read_text_file(root / "config.json", "config.json");
  if (!config_text) {
    return std::unexpected(config_text.error());
  }
  auto cfg = parse_text_config(*config_text);
  if (!cfg) {
    return std::unexpected(cfg.error());
  }
  if (auto st = validate_architecture(*cfg); !st) {
    return std::unexpected(st.error());
  }
  ckpt.config = *cfg;

  // Opening a checkpoint only inspects its metadata and safetensors headers.
  // In particular, do not stream every shard here: normal compiler tests open
  // the authority to inspect its schema and must not rehash its payload.
  auto config_hash = hash_file(root / "config.json", "config.json");
  if (!config_hash) {
    return std::unexpected(config_hash.error());
  }
  ckpt.config_hash = *config_hash;
  auto index_hash = hash_file(root / "model.safetensors.index.json",
                              "model.safetensors.index.json");
  if (!index_hash) {
    return std::unexpected(index_hash.error());
  }
  ckpt.source_hash = *index_hash;
  auto tokenizer_path = root / "tokenizer.json";
  if (!std::filesystem::exists(tokenizer_path)) {
    tokenizer_path = root / "tokenizer_config.json";
  }
  auto tokenizer_hash = hash_file(tokenizer_path, "tokenizer");
  if (!tokenizer_hash) {
    return std::unexpected(tokenizer_hash.error());
  }
  ckpt.tokenizer_hash = *tokenizer_hash;

  auto index_text =
      read_text_file(root / "model.safetensors.index.json", "index");
  if (!index_text) {
    return std::unexpected(index_text.error());
  }
  auto index = parse_json(*index_text, "model.safetensors.index.json");
  if (!index) {
    return std::unexpected(index.error());
  }
  auto const* weight_map = index->find("weight_map");
  if (weight_map == nullptr || !weight_map->is_object()) {
    return std::unexpected(make_error(CompilerErrorCode::InvalidConfig,
                                      "weight_map", "index has no weight_map"));
  }

  std::unordered_map<std::string, std::string> map;
  for (auto const& [name, shard] : weight_map->as_object()) {
    if (!shard.is_string()) {
      return std::unexpected(make_error(CompilerErrorCode::InvalidConfig, name,
                                        "weight_map value is not a string"));
    }
    map.emplace(name, shard.as_string());
  }

  std::vector<SourceTensor> tensors;
  tensors.reserve(map.size());
  for (auto const& [shard_name, _] : [&] {
         std::map<std::string, int> unique;
         for (auto const& [n, s] : map) {
           unique.emplace(s, 0);
         }
         return unique;
       }()) {
    auto path = root / shard_name;
    auto mapped = MappedShard::open(path);
    if (!mapped) {
      return std::unexpected(mapped.error());
    }
    auto header = parse_shard_header(mapped->bytes(), shard_name);
    if (!header) {
      return std::unexpected(header.error());
    }
    OpenShard open{};
    open.path = path;
    auto header_len = read_u64_le(mapped->bytes(), 0, shard_name);
    if (!header_len) {
      return std::unexpected(header_len.error());
    }
    open.header_bytes = *header_len;
    open.payload_base = 8 + *header_len;
    open.file_size = mapped->bytes().size();
    ckpt.shards.emplace(shard_name, open);
    for (auto& t : *header) {
      auto it = map.find(t.name);
      if (it == map.end()) {
        return std::unexpected(make_error(
            CompilerErrorCode::ExtraTensor, t.name,
            "shard header name is absent from weight_map"));
      }
      if (it->second != shard_name) {
        return std::unexpected(make_error(
            CompilerErrorCode::ArchitectureMismatch, t.name,
            "weight_map shard does not match header location"));
      }
      tensors.push_back(std::move(t));
    }
  }

  if (tensors.size() != map.size()) {
    return std::unexpected(make_error(CompilerErrorCode::MissingTensor, "index",
                                      "weight_map is not covered by shard headers"));
  }
  if (ckpt.shards.size() != kShards && tensors.size() == kCheckpointTensors) {
    return std::unexpected(make_error(CompilerErrorCode::ArchitectureMismatch,
                                      "shards", "expected 18 safetensor shards"));
  }

  auto classified = classify_source_tensors(std::move(tensors));
  if (!classified) {
    return std::unexpected(classified.error());
  }
  classified->shard_count = static_cast<std::uint32_t>(ckpt.shards.size());
  ckpt.classified = std::move(*classified);
  return ckpt;
}

}  // namespace qw38::compiler
