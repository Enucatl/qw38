#include "compiler/compiler.hpp"
#include "format/format.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

using qw38::compiler::classify_source_tensors;
using qw38::compiler::CompilerErrorCode;
using qw38::compiler::expand_identity_table;
using qw38::compiler::family_identity_table;
using qw38::compiler::is_vision_tensor;
using qw38::compiler::kIncludedTensors;
using qw38::compiler::parse_json_u64;
using qw38::compiler::parse_safetensors_shape;
using qw38::compiler::SourceClass;
using qw38::compiler::SourceTensor;
using qw38::compiler::SyntheticTensor;
using qw38::compiler::TensorFamily;

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

}  // namespace

int main() {
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
  auto classified = classify_source_tensors(headers);
  if (!classified) {
    fail(qw38::compiler::error_message(classified.error()));
  } else {
    expect(classified->included.size() == kIncludedTensors, "classified 866");
    expect(classified->vision_excluded == 0, "no vision in language-only set");
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
    auto with_vision = headers;
    with_vision.push_back(
        make_source("model.visual.patch_embed.proj.weight", {4}, off));
    auto got = classify_source_tensors(with_vision);
    expect(static_cast<bool>(got) && got->vision_excluded == 1,
           "vision tensors are excluded");
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
