#include "compiler/compiler.hpp"
#include "format/format.hpp"

#include <cmath>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using qw38::compiler::classify_source_tensors;
using qw38::compiler::CompilerErrorCode;
using qw38::compiler::expand_identity_table;
using qw38::compiler::expand_vision_inventory;
using qw38::compiler::family_identity_table;
using qw38::compiler::is_vision_tensor;
using qw38::compiler::kIncludedTensors;
using qw38::compiler::parse_json_u64;
using qw38::compiler::parse_safetensors_shape;
using qw38::compiler::parse_shard_header;
using qw38::compiler::parse_text_config;
using qw38::compiler::SourceClass;
using qw38::compiler::SourceTensor;
using qw38::compiler::SyntheticTensor;
using qw38::compiler::TensorFamily;
using qw38::compiler::validate_architecture;

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

SourceTensor make_source(std::string name, std::vector<std::uint64_t> shape,
                         std::uint64_t offset, std::string dtype = "BF16") {
  SourceTensor t{};
  t.name = std::move(name);
  t.dtype = std::move(dtype);
  t.shape = std::move(shape);
  t.shard = "model-00001-of-00018.safetensors";
  t.data_offset = offset;
  std::uint64_t n = 1;
  for (auto d : t.shape) {
    n *= d;
  }
  t.nbytes = n * 2;
  return t;
}

std::vector<SourceTensor> complete_language_mtp(std::uint64_t* offset) {
  auto expected = expand_identity_table();
  std::vector<SourceTensor> out;
  out.reserve(expected.size());
  for (auto const& e : expected) {
    std::vector<std::uint64_t> shape(e.shape.dims.begin(),
                                     e.shape.dims.begin() + e.shape.rank);
    auto t = make_source(e.name, std::move(shape), *offset);
    *offset += t.nbytes;
    out.push_back(std::move(t));
  }
  return out;
}

void append_vision_inventory(std::vector<SourceTensor>* tensors,
                             std::uint64_t* offset) {
  for (auto const& e : expand_vision_inventory()) {
    std::vector<std::uint64_t> shape(e.shape.dims.begin(),
                                     e.shape.dims.begin() + e.shape.rank);
    auto t = make_source(e.name, std::move(shape), *offset);
    t.shard = e.shard;
    *offset += t.nbytes;
    tensors->push_back(std::move(t));
  }
}

std::string valid_config_json() {
  std::ostringstream out;
  out << R"({"architectures":["Qwen3_5ForConditionalGeneration"],)"
      << R"("model_type":"qwen3_5","tie_word_embeddings":false,)"
      << R"("text_config":{"attention_bias":false,"attention_dropout":0.0,)"
      << R"("attn_output_gate":true,"dtype":"bfloat16",)"
      << R"("full_attention_interval":4,"head_dim":256,"hidden_act":"silu",)"
      << R"("hidden_size":5120,"intermediate_size":17408,)"
      << R"("linear_conv_kernel_dim":4,"linear_key_head_dim":128,)"
      << R"("linear_num_key_heads":16,"linear_num_value_heads":48,)"
      << R"("linear_value_head_dim":128,"mamba_ssm_dtype":"float32",)"
      << R"("max_position_embeddings":262144,"model_type":"qwen3_5_text",)"
      << R"("mtp_num_hidden_layers":1,"mtp_use_dedicated_embeddings":false,)"
      << R"("num_attention_heads":24,"num_hidden_layers":64,)"
      << R"("num_key_value_heads":4,"output_gate_type":"swish",)"
      << R"("partial_rotary_factor":0.25,"rms_norm_eps":1e-6,)"
      << R"("rope_parameters":{"mrope_interleaved":true,)"
      << R"("mrope_section":[11,11,10],"partial_rotary_factor":0.25,)"
      << R"("rope_theta":10000000,"rope_type":"default"},)"
      << R"("tie_word_embeddings":false,"use_cache":true,"vocab_size":248320,)"
      << R"("layer_types":[)";
  for (std::uint32_t layer = 0; layer < 64; ++layer) {
    if (layer != 0) {
      out << ',';
    }
    out << (layer % 4 == 3 ? R"("full_attention")"
                           : R"("linear_attention")");
  }
  out << "]}}";
  return out.str();
}

std::string replace_one(std::string input, std::string_view from,
                        std::string_view to) {
  auto const at = input.find(from);
  if (at == std::string::npos ||
      input.find(from, at + from.size()) != std::string::npos) {
    fail(std::string("mutation target is not unique: ") + std::string(from));
    return input;
  }
  input.replace(at, from.size(), to);
  return input;
}

std::vector<std::byte> safetensors_file(std::string_view header,
                                        std::size_t payload_bytes) {
  std::vector<std::byte> file(8 + header.size() + payload_bytes);
  auto const size = static_cast<std::uint64_t>(header.size());
  for (std::size_t i = 0; i < 8; ++i) {
    file[i] = static_cast<std::byte>((size >> (8 * i)) & 0xffU);
  }
  std::memcpy(file.data() + 8, header.data(), header.size());
  return file;
}

}  // namespace

int main() {
  {
    auto const baseline_text = valid_config_json();
    auto baseline = parse_text_config(baseline_text);
    expect(static_cast<bool>(baseline), "complete architecture config parses");
    if (baseline) {
      auto status = validate_architecture(*baseline);
      expect(static_cast<bool>(status), "complete architecture config validates");
    }

    struct Mutation {
      std::string_view from;
      std::string_view to;
      std::string_view field;
    };
    std::vector<Mutation> const mutations{
        {R"("Qwen3_5ForConditionalGeneration")", R"("Qwen3ForCausalLM")",
         "architectures"},
        {R"("model_type":"qwen3_5","tie_word_embeddings")",
         R"("model_type":"qwen3","tie_word_embeddings")", "model_type"},
        {R"("model_type":"qwen3_5_text")", R"("model_type":"qwen3_text")",
         "text_config.model_type"},
        {R"("model_type":"qwen3_5","tie_word_embeddings":false)",
         R"("model_type":"qwen3_5","tie_word_embeddings":true)",
         "tie_word_embeddings"},
        {R"("tie_word_embeddings":false,"use_cache")",
         R"("tie_word_embeddings":true,"use_cache")", "tie_word_embeddings"},
        {R"("attention_bias":false)", R"("attention_bias":true)",
         "attention_bias"},
        {R"("attention_dropout":0.0)", R"("attention_dropout":0.1)",
         "attention_dropout"},
        {R"("attn_output_gate":true)", R"("attn_output_gate":false)",
         "attn_output_gate"},
        {R"("dtype":"bfloat16")", R"("dtype":"float16")", "dtype"},
        {R"("full_attention_interval":4)", R"("full_attention_interval":8)",
         "full_attention_interval"},
        {R"("head_dim":256)", R"("head_dim":128)", "head_dim"},
        {R"("hidden_act":"silu")", R"("hidden_act":"gelu")", "hidden_act"},
        {R"("hidden_size":5120)", R"("hidden_size":4096)", "hidden_size"},
        {R"("intermediate_size":17408)", R"("intermediate_size":16384)",
         "intermediate_size"},
        {R"("linear_conv_kernel_dim":4)", R"("linear_conv_kernel_dim":3)",
         "linear_conv_kernel_dim"},
        {R"("linear_key_head_dim":128)", R"("linear_key_head_dim":64)",
         "linear_key_head_dim"},
        {R"("linear_num_key_heads":16)", R"("linear_num_key_heads":8)",
         "linear_num_key_heads"},
        {R"("linear_num_value_heads":48)", R"("linear_num_value_heads":32)",
         "linear_num_value_heads"},
        {R"("linear_value_head_dim":128)", R"("linear_value_head_dim":64)",
         "linear_value_head_dim"},
        {R"("mamba_ssm_dtype":"float32")",
         R"("mamba_ssm_dtype":"bfloat16")", "mamba_ssm_dtype"},
        {R"("max_position_embeddings":262144)",
         R"("max_position_embeddings":131072)", "max_position_embeddings"},
        {R"("mtp_num_hidden_layers":1)", R"("mtp_num_hidden_layers":2)",
         "mtp_num_hidden_layers"},
        {R"("mtp_use_dedicated_embeddings":false)",
         R"("mtp_use_dedicated_embeddings":true)",
         "mtp_use_dedicated_embeddings"},
        {R"("num_attention_heads":24)", R"("num_attention_heads":16)",
         "num_attention_heads"},
        {R"("num_hidden_layers":64)", R"("num_hidden_layers":63)",
         "num_hidden_layers"},
        {R"("num_key_value_heads":4)", R"("num_key_value_heads":8)",
         "num_key_value_heads"},
        {R"("output_gate_type":"swish")",
         R"("output_gate_type":"sigmoid")", "output_gate_type"},
        {R"("partial_rotary_factor":0.25,"rms_norm_eps")",
         R"("partial_rotary_factor":0.5,"rms_norm_eps")",
         "partial_rotary_factor"},
        {R"("rms_norm_eps":1e-6)", R"("rms_norm_eps":1e-5)",
         "rms_norm_eps"},
        {R"("mrope_interleaved":true)", R"("mrope_interleaved":false)",
         "mrope_interleaved"},
        {R"("mrope_section":[11,11,10])", R"("mrope_section":[10,11,11])",
         "mrope_section"},
        {R"("mrope_section":[11,11,10],"partial_rotary_factor":0.25)",
         R"("mrope_section":[11,11,10],"partial_rotary_factor":0.5)",
         "partial_rotary_factor"},
        {R"("rope_theta":10000000)", R"("rope_theta":10000)", "rope_theta"},
        {R"("rope_type":"default")", R"("rope_type":"linear")", "rope_type"},
        {R"("use_cache":true)", R"("use_cache":false)", "use_cache"},
        {R"("vocab_size":248320)", R"("vocab_size":248321)", "vocab_size"},
        {R"("layer_types":["linear_attention","linear_attention","linear_attention")",
         R"("layer_types":["full_attention","linear_attention","linear_attention")",
         "layer_types"},
    };
    for (auto const& mutation : mutations) {
      auto changed = replace_one(baseline_text, mutation.from, mutation.to);
      auto parsed = parse_text_config(changed);
      if (!parsed) {
        fail(std::string("mutation did not parse: ") +
             qw38::compiler::error_message(parsed.error()));
        continue;
      }
      auto status = validate_architecture(*parsed);
      expect(!status &&
                 status.error().code == CompilerErrorCode::ArchitectureMismatch &&
                 status.error().field == mutation.field,
             std::string("typed config rejection: ") +
                 std::string(mutation.field));
    }
  }

  {
    qw38::compiler::Json::Array rank8(8, qw38::compiler::Json{1.0});
    auto shape = parse_safetensors_shape(qw38::compiler::Json{rank8}, "shape");
    expect(shape && shape->size() == 8, "safetensors rank 8 is valid");

    qw38::compiler::Json::Array rank9(9, qw38::compiler::Json{1.0});
    shape = parse_safetensors_shape(qw38::compiler::Json{rank9}, "shape");
    expect(!shape && shape.error().code == CompilerErrorCode::ShapeMismatch,
           "safetensors rank 9 is rejected");
    shape = parse_safetensors_shape(
        qw38::compiler::Json{qw38::compiler::Json::Array{}}, "shape");
    expect(!shape && shape.error().code == CompilerErrorCode::ShapeMismatch,
           "safetensors rank 0 is rejected");
    shape = parse_safetensors_shape(
        qw38::compiler::Json{
            qw38::compiler::Json::Array{qw38::compiler::Json{0.0}}},
        "shape");
    expect(shape && (*shape)[0] == 0,
           "safetensors zero dimension is parsed without conversion UB");

    for (double invalid :
         {1.5, -1.0, std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::infinity()}) {
      auto value = parse_json_u64(qw38::compiler::Json{invalid}, "dimension");
      expect(!value && value.error().code == CompilerErrorCode::InvalidJson,
             "invalid JSON integer is rejected");
    }
    auto too_large =
        parse_json_u64(qw38::compiler::Json{std::ldexp(1.0, 64)}, "dimension");
    expect(!too_large && too_large.error().code == CompilerErrorCode::InvalidJson,
           "unrepresentable JSON integer is rejected");
  }

  {
    auto parse_header = [](std::string_view offsets, std::size_t payload = 2) {
      std::string const header =
          R"({"tensor":{"dtype":"BF16","shape":[1],"data_offsets":)" +
          std::string(offsets) + "}}";
      auto file = safetensors_file(header, payload);
      return parse_shard_header(file, "fixture.safetensors");
    };

    auto valid = parse_header("[0,2]");
    expect(valid && valid->size() == 1 && valid->front().nbytes == 2,
           "valid safetensors offsets parse");

    for (std::string_view offsets :
         {"[\"0\",2]", "[false,2]", "[null,2]", "[{},2]", "[[0],2]",
          "[0.5,2]", "[-1,2]", "[0,1.5]", "[0,-1]",
          "[0,18446744073709551616]"}) {
      auto malformed = parse_header(offsets);
      expect(!malformed &&
                 malformed.error().code == CompilerErrorCode::InvalidJson,
             "malformed safetensors offset is a typed error");
    }

    auto reversed = parse_header("[2,0]");
    expect(!reversed &&
               reversed.error().code == CompilerErrorCode::ShapeMismatch,
           "reversed safetensors interval is rejected");
    auto out_of_file = parse_header("[0,3]");
    expect(!out_of_file &&
               out_of_file.error().code == CompilerErrorCode::ShapeMismatch,
           "out-of-file safetensors interval is rejected");
    auto huge_interval = parse_header("[0,18446744073709549568]");
    expect(!huge_interval &&
               huge_interval.error().code == CompilerErrorCode::ShapeMismatch,
           "payload interval arithmetic cannot overflow");
  }

  auto expected = expand_identity_table();
  expect(expected.size() == kIncludedTensors, "identity table expands to 866");
  expect(family_identity_table().size() >= 38, "family table is encoded");

  std::uint32_t language = 0;
  std::uint32_t mtp = 0;
  for (auto const& e : expected) {
    if (e.source_class == SourceClass::Language) {
      ++language;
    } else if (e.source_class == SourceClass::MtpRetainedDisabled) {
      ++mtp;
    }
  }
  expect(language == 851, "851 language tensors");
  expect(mtp == 15, "15 retained MTP tensors");
  expect(is_vision_tensor("model.visual.blocks.0.attn.qkv.weight"),
         "vision prefix");
  expect(!is_vision_tensor("model.language_model.norm.weight"),
         "language is not vision");

  std::uint64_t off = 0;
  auto headers = complete_language_mtp(&off);
  append_vision_inventory(&headers, &off);
  auto classified = classify_source_tensors(headers);
  if (!classified) {
    fail(qw38::compiler::error_message(classified.error()));
  } else {
    expect(classified->included.size() == kIncludedTensors, "classified 866");
    expect(classified->vision_excluded == 333, "exact vision inventory excluded");
    bool saw_embed = false;
    bool saw_conv = false;
    bool saw_mtp = false;
    for (auto const& t : classified->included) {
      if (t.expected.family == TensorFamily::Embed) {
        saw_embed = true;
        expect(t.expected.layout ==
                   qw38::format::PhysicalLayoutId::CudaBf16RowMajorV0,
               "embed row-major");
      }
      if (t.expected.family == TensorFamily::LinearAttnConv1d) {
        saw_conv = true;
        expect(t.expected.layout ==
                   qw38::format::PhysicalLayoutId::CudaBf16TapMajorV0,
               "conv tap-major");
      }
      if (t.expected.source_class == SourceClass::MtpRetainedDisabled) {
        saw_mtp = true;
      }
    }
    expect(saw_embed && saw_conv && saw_mtp, "families classified");
  }

  {
    auto missing = headers;
    missing.pop_back();
    auto got = classify_source_tensors(missing);
    expect(!got && got.error().code == CompilerErrorCode::MissingTensor,
           "missing tensor");
  }
  {
    auto missing = headers;
    auto const it = std::find_if(missing.begin(), missing.end(), [](auto const& t) {
      return t.name == "model.visual.pos_embed.weight";
    });
    missing.erase(it);
    auto got = classify_source_tensors(missing);
    expect(!got && got.error().code == CompilerErrorCode::MissingTensor,
           "missing vision tensor");
  }
  {
    auto extra = headers;
    extra.push_back(make_source("model.visual.unrecognized.weight", {1}, off));
    auto got = classify_source_tensors(extra);
    expect(!got && got.error().code == CompilerErrorCode::ExtraTensor,
           "extra vision tensor");
  }
  {
    auto bad = headers;
    auto const it = std::find_if(bad.begin(), bad.end(), [](auto const& t) {
      return t.name == "model.visual.pos_embed.weight";
    });
    it->shape[0] += 1;
    auto got = classify_source_tensors(bad);
    expect(!got && got.error().code == CompilerErrorCode::ShapeMismatch,
           "vision shape mismatch");
  }
  {
    auto bad = headers;
    auto const it = std::find_if(bad.begin(), bad.end(), [](auto const& t) {
      return t.name == "model.visual.pos_embed.weight";
    });
    it->dtype = "F32";
    auto got = classify_source_tensors(bad);
    expect(!got && got.error().code == CompilerErrorCode::DtypeMismatch,
           "vision dtype mismatch");
  }
  {
    auto extra = headers;
    extra.push_back(make_source("unexpected.weight", {4}, off));
    auto got = classify_source_tensors(extra);
    expect(!got && got.error().code == CompilerErrorCode::ExtraTensor,
           "extra tensor");
  }
  {
    auto dup = headers;
    dup.push_back(headers.front());
    auto got = classify_source_tensors(dup);
    expect(!got && got.error().code == CompilerErrorCode::DuplicateTensor,
           "duplicate tensor");
  }
  {
    auto bad = headers;
    bad[0].shape[0] += 1;
    auto got = classify_source_tensors(bad);
    expect(!got && got.error().code == CompilerErrorCode::ShapeMismatch,
           "shape mismatch");
  }
  {
    auto bad = headers;
    bad[0].shape.assign(9, 1);
    auto got = classify_source_tensors(bad);
    expect(!got && got.error().code == CompilerErrorCode::ShapeMismatch,
           "rank-9 source tensor is rejected");
  }
  {
    auto bad = headers;
    bad[0].shape[0] = 0;
    auto got = classify_source_tensors(bad);
    expect(!got && got.error().code == CompilerErrorCode::ShapeMismatch,
           "zero source extent is rejected");
  }
  {
    auto bad = headers;
    bad[0].shape = {std::numeric_limits<std::uint64_t>::max(), 2};
    auto got = classify_source_tensors(bad);
    expect(!got && got.error().code == CompilerErrorCode::Unrepresentable,
           "source dimension product overflow is rejected");
  }
  {
    auto bad = headers;
    bad[0].dtype = "F32";
    auto got = classify_source_tensors(bad);
    expect(!got && got.error().code == CompilerErrorCode::DtypeMismatch,
           "dtype mismatch");
  }
  {
    auto shared = headers;
    shared[1].shard = shared[0].shard;
    shared[1].data_offset = shared[0].data_offset;
    shared[1].nbytes = shared[0].nbytes;
    auto got = classify_source_tensors(shared);
    expect(!got && got.error().code == CompilerErrorCode::UnexpectedSharing,
           "unexpected sharing");
  }
  {
    auto overlapping = headers;
    overlapping[1].shard = overlapping[0].shard;
    overlapping[1].data_offset = overlapping[0].data_offset + 2;
    auto got = classify_source_tensors(overlapping);
    expect(!got && got.error().code == CompilerErrorCode::UnexpectedSharing,
           "partial source tensor overlap is rejected");
  }
  {
    auto contained = headers;
    contained[1].shard = contained[0].shard;
    contained[1].data_offset = contained[0].data_offset + 2;
    contained[1].nbytes = 2;
    auto got = classify_source_tensors(contained);
    expect(!got && got.error().code == CompilerErrorCode::UnexpectedSharing,
           "contained source tensor overlap is rejected");
  }
  {
    auto vision_shared = headers;
    auto first = std::find_if(vision_shared.begin(), vision_shared.end(),
                              [](auto const& t) {
                                return t.name == "model.visual.blocks.0.norm1.bias";
                              });
    auto second = std::find_if(vision_shared.begin(), vision_shared.end(),
                               [](auto const& t) {
                                 return t.name == "model.visual.blocks.0.norm1.weight";
                               });
    second->data_offset = first->data_offset;
    second->nbytes = first->nbytes;
    auto got = classify_source_tensors(vision_shared);
    expect(got && got->vision_excluded == 333,
           "explicit exact sharing for excluded vision tensors is valid");
  }
  {
    auto overflow = headers;
    overflow[0].data_offset = std::numeric_limits<std::uint64_t>::max();
    overflow[0].nbytes = 2;
    auto got = classify_source_tensors(overflow);
    expect(!got && got.error().code == CompilerErrorCode::Unrepresentable,
           "source interval overflow is rejected");
  }
  {
    SyntheticTensor malformed{};
    malformed.expected = expected.front();
    malformed.expected.shape.rank = 9;
    auto got = qw38::compiler::compile_synthetic(
        std::filesystem::path{"/tmp/qw38-malformed-rank.qw38"}, {}, {}, {},
        {.ident = "test"}, {std::move(malformed)});
    expect(!got && got.error().code == CompilerErrorCode::ShapeMismatch,
           "compile entry point rejects malformed fixed-array rank");
  }

  if (g_failures != 0) {
    std::cerr << g_failures << " failures\n";
    return 1;
  }
  return 0;
}
