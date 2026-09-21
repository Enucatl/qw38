#include "compiler/compiler.hpp"
#include "format/format.hpp"
#include "format_reader_support.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

using qw38::compiler::build_identity_schema;
using qw38::compiler::ClassifiedCheckpoint;
using qw38::compiler::compile_synthetic;
using qw38::format::CompilerRevision;
using qw38::compiler::conv_from_tap_major;
using qw38::compiler::count_instances;
using qw38::compiler::expand_identity_table;
using qw38::compiler::generate_rope_inv_freq;
using qw38::compiler::kCompilerIdent;
using qw38::compiler::kIncludedTensors;
using qw38::compiler::kLanguageInstances;
using qw38::compiler::kMtpInstances;
using qw38::compiler::kRopeInvFreqName;
using qw38::compiler::kShards;
using qw38::compiler::kVisionTensors;
using qw38::compiler::open_checkpoint;
using qw38::compiler::row_major_from_tile_nk;
using qw38::compiler::SourceClass;
using qw38::compiler::SourceTensor;
using qw38::compiler::SyntheticTensor;
using qw38::compiler::TensorFamily;
using qw38::format::Artifact;
using qw38::format::Hash256;
using qw38::format::kNoLayerIndex;
using qw38::format::PhysicalLayoutId;
using qw38::format::SemanticScope;
using qw38::format::SharedBindingRole;
using qw38::format::test::read_all;
using qw38::format::test::ScratchDir;

#ifndef QW38_SOURCE_DIR
#define QW38_SOURCE_DIR "."
#endif

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

Hash256 hash_seed(std::uint8_t seed) {
  Hash256 h{};
  for (std::size_t i = 0; i < h.bytes.size(); ++i) {
    h.bytes[i] = static_cast<std::uint8_t>(seed + i);
  }
  return h;
}

std::vector<std::byte> pattern(std::uint64_t elems, std::uint16_t seed) {
  std::vector<std::byte> out(elems * 2);
  for (std::uint64_t i = 0; i < elems; ++i) {
    std::uint16_t bits =
        static_cast<std::uint16_t>(0x3C00 + ((seed + i) & 0x007F));
    std::memcpy(out.data() + i * 2, &bits, 2);
  }
  return out;
}

qw38::compiler::ExpectedTensor find_expected(TensorFamily family) {
  for (auto const& e : expand_identity_table()) {
    if (e.family == family) {
      return e;
    }
  }
  return {};
}

SyntheticTensor synth_from(qw38::compiler::ExpectedTensor exp,
                           std::uint16_t seed) {
  std::uint64_t n = 1;
  for (std::uint8_t i = 0; i < exp.shape.rank; ++i) {
    n *= exp.shape.dims[i];
  }
  SyntheticTensor t{};
  t.expected = std::move(exp);
  t.bytes = pattern(n, seed);
  return t;
}

SyntheticTensor small_matrix(qw38::compiler::ExpectedTensor exp, std::uint64_t n,
                             std::uint64_t k, std::uint16_t seed) {
  exp.shape.rank = 2;
  exp.shape.dims = {n, k, 0};
  SyntheticTensor t{};
  t.expected = std::move(exp);
  t.bytes = pattern(n * k, seed);
  return t;
}

}  // namespace

int main() {
  auto expected = expand_identity_table();
  ClassifiedCheckpoint classified{};
  classified.included.reserve(expected.size());
  std::uint64_t off = 0;
  for (auto const& e : expected) {
    SourceTensor src{};
    src.name = e.name;
    src.dtype = "BF16";
    src.shape.assign(e.shape.dims.begin(), e.shape.dims.begin() + e.shape.rank);
    src.shard = "model-00001-of-00018.safetensors";
    src.data_offset = off;
    std::uint64_t n = 1;
    for (auto d : src.shape) {
      n *= d;
    }
    src.nbytes = n * 2;
    off += src.nbytes;
    classified.included.push_back(
        {.expected = e, .source = std::move(src)});
  }

  auto schema = build_identity_schema(classified, hash_seed(1), hash_seed(2),
                                      hash_seed(3),
                                      CompilerRevision{.ident = kCompilerIdent,
                                                       .major = 0,
                                                       .minor = 1,
                                                       .patch = 0});
  if (!schema) {
    fail(qw38::compiler::error_message(schema.error()));
  } else {
    expect(schema->scope == SemanticScope::LanguagePlusMtpDescriptors,
           "MTP descriptors retained");
    expect(count_instances(schema->graph_bindings, SourceClass::Language) ==
               kLanguageInstances,
           "130 language instances");
    expect(count_instances(schema->graph_bindings,
                           SourceClass::MtpRetainedDisabled) == kMtpInstances,
           "5 retained-disabled MTP instances");
    expect(schema->shared_bindings.size() == 2, "embed and lm_head aliases");
    expect(schema->shared_bindings[0].role ==
                   SharedBindingRole::MtpEmbeddingAlias &&
               schema->shared_bindings[1].role ==
                   SharedBindingRole::MtpLmHeadAlias,
           "MTP alias roles");
    expect(schema->state.size() == 3, "GDN/conv/KV state schema");
    expect(schema->scratch.size() == 7, "language scratch schema");
    bool saw_rope = false;
    for (auto const& t : schema->tensors) {
      if (t.logical_name == kRopeInvFreqName) {
        saw_rope = true;
        expect(t.layout == PhysicalLayoutId::CudaFp32VectorV0, "RoPE FP32 vector");
      }
    }
    expect(saw_rope, "generated RoPE tensor");
  }

  ScratchDir dir{"qw38-compiler-int"};
  auto dest1 = dir.path() / "a.qw38";
  auto dest2 = dir.path() / "b.qw38";
  auto embed = small_matrix(find_expected(TensorFamily::Embed), 8, 8, 1);
  embed.expected.layout = PhysicalLayoutId::CudaBf16RowMajorV0;
  auto head = small_matrix(find_expected(TensorFamily::LmHead), 8, 256, 2);
  auto dense = small_matrix(find_expected(TensorFamily::LinearAttnInProjQkv), 8,
                            256, 3);
  auto conv = synth_from(find_expected(TensorFamily::LinearAttnConv1d), 4);
  conv.expected.shape = {.rank = 3, .dims = {32, 1, 4}};
  conv.bytes = pattern(32 * 4, 4);
  auto vec = synth_from(find_expected(TensorFamily::LinearAttnALog), 5);

  CompilerRevision rev{.ident = kCompilerIdent, .major = 0, .minor = 1, .patch = 0};
  auto hashes = std::array{hash_seed(0x11), hash_seed(0x22), hash_seed(0x33)};
  std::vector<SyntheticTensor> fixture{embed, head, dense, conv, vec};

  auto first = compile_synthetic(dest1, hashes[0], hashes[1], hashes[2], rev,
                                 fixture);
  auto second = compile_synthetic(dest2, hashes[0], hashes[1], hashes[2], rev,
                                  fixture);
  if (!first || !second) {
    fail("synthetic compile");
    if (!first) {
      fail(qw38::compiler::error_message(first.error()));
    }
    if (!second) {
      fail(qw38::compiler::error_message(second.error()));
    }
  } else {
    auto a = read_all(dest1);
    auto b = read_all(dest2);
    expect(a == b, "repeated compile is byte-identical");
    auto art = Artifact::open(dest1);
    if (!art) {
      fail(qw38::format::error_message(art.error()));
    } else {
      expect(art->scope() == SemanticScope::LanguagePlusMtpDescriptors,
             "reader scope");
      expect(art->state().size() == 3 && art->scratch().size() == 7,
             "reader state/scratch");
      expect(art->shared_bindings().size() == 2, "reader shared bindings");
      auto rope = art->payload(kRopeInvFreqName);
      auto want = generate_rope_inv_freq();
      expect(static_cast<bool>(rope) &&
                 std::equal(rope->begin(), rope->end(), want.begin()),
             "artifact RoPE matches generator");
      auto conv_payload = art->payload(conv.expected.name);
      if (!conv_payload) {
        fail("conv payload");
      } else {
        auto restored = conv_from_tap_major(*conv_payload, 32, 4);
        expect(static_cast<bool>(restored) && *restored == conv.bytes,
               "fixture conv reconstructs exactly");
      }
      auto dense_payload = art->payload(dense.expected.name);
      if (!dense_payload) {
        fail("dense payload");
      } else {
        auto restored = row_major_from_tile_nk(*dense_payload, 8, 256);
        expect(static_cast<bool>(restored) && *restored == dense.bytes,
               "fixture dense reconstructs exactly");
      }
      auto vec_payload = art->payload(vec.expected.name);
      expect(static_cast<bool>(vec_payload) &&
                 vec_payload->size() == vec.bytes.size() &&
                 std::equal(vec_payload->begin(), vec_payload->end(),
                            vec.bytes.begin()),
             "fixture vector is identity");
    }
  }

  std::filesystem::path authority =
      std::filesystem::path(QW38_SOURCE_DIR) /
      ".cache/authorities/qwen3.8-27b-transformers";
  if (std::filesystem::exists(authority / "config.json") &&
      std::filesystem::exists(authority / "model.safetensors.index.json")) {
    auto ckpt = open_checkpoint(authority);
    if (!ckpt) {
      fail(std::string("checkpoint: ") +
           qw38::compiler::error_message(ckpt.error()));
    } else {
      expect(ckpt->classified.included.size() == kIncludedTensors,
             "authoritative occupancy 866");
      expect(ckpt->classified.vision_excluded == kVisionTensors,
             "vision excluded 333");
      expect(ckpt->shards.size() == kShards, "18 shards");
      auto live = build_identity_schema(ckpt->classified, ckpt->source_hash,
                                        ckpt->config_hash, ckpt->tokenizer_hash,
                                        rev);
      if (!live) {
        fail(qw38::compiler::error_message(live.error()));
      } else {
        expect(count_instances(live->graph_bindings, SourceClass::Language) ==
                   kLanguageInstances,
               "checkpoint schema has 130 language instances");
        expect(count_instances(live->graph_bindings,
                               SourceClass::MtpRetainedDisabled) ==
                   kMtpInstances,
               "checkpoint schema retains disabled MTP");
      }

      auto load = [&](TensorFamily family) -> std::vector<std::byte> {
        for (auto const& item : ckpt->classified.included) {
          if (item.expected.family != family) {
            continue;
          }
          auto mapped = qw38::compiler::MappedShard::open(
              ckpt->shards.at(item.source.shard).path);
          if (!mapped) {
            fail(qw38::compiler::error_message(mapped.error()));
            return {};
          }
          auto bytes = mapped->tensor_bytes(item.source);
          if (!bytes) {
            fail(qw38::compiler::error_message(bytes.error()));
            return {};
          }
          return std::vector<std::byte>(bytes->begin(), bytes->end());
        }
        return {};
      };
      auto real_alog = load(TensorFamily::LinearAttnALog);
      auto real_conv = load(TensorFamily::LinearAttnConv1d);
      auto real_ab = load(TensorFamily::LinearAttnInProjA);
      expect(!real_alog.empty() && !real_conv.empty() && !real_ab.empty(),
             "sampled source tensors");
      if (!real_alog.empty() && !real_conv.empty() && !real_ab.empty()) {
        auto s_alog = synth_from(find_expected(TensorFamily::LinearAttnALog), 0);
        s_alog.bytes = std::move(real_alog);
        auto s_conv = synth_from(find_expected(TensorFamily::LinearAttnConv1d), 0);
        s_conv.bytes = std::move(real_conv);
        auto s_ab = synth_from(find_expected(TensorFamily::LinearAttnInProjA), 0);
        s_ab.bytes = std::move(real_ab);
        auto dest = dir.path() / "sampled.qw38";
        std::vector<SyntheticTensor> sampled{embed, head, s_alog, s_conv, s_ab};
        auto compiled = compile_synthetic(dest, ckpt->source_hash,
                                          ckpt->config_hash,
                                          ckpt->tokenizer_hash, rev, sampled);
        if (!compiled) {
          fail(qw38::compiler::error_message(compiled.error()));
        } else {
          auto art = Artifact::open(dest);
          if (!art) {
            fail(qw38::format::error_message(art.error()));
          } else {
            auto p_alog = art->payload(s_alog.expected.name);
            expect(static_cast<bool>(p_alog) &&
                       p_alog->size() == s_alog.bytes.size() &&
                       std::equal(p_alog->begin(), p_alog->end(),
                                  s_alog.bytes.begin()),
                   "sampled A_log identity");
            auto p_conv = art->payload(s_conv.expected.name);
            if (!p_conv) {
              fail("sampled conv payload");
            } else {
              auto conv_back = conv_from_tap_major(*p_conv, 10240, 4);
              expect(static_cast<bool>(conv_back) && *conv_back == s_conv.bytes,
                     "sampled conv reconstructs exactly");
            }
            auto p_ab = art->payload(s_ab.expected.name);
            if (!p_ab) {
              fail("sampled in_proj_a payload");
            } else {
              auto ab_back = row_major_from_tile_nk(*p_ab, 48, 5120);
              expect(static_cast<bool>(ab_back) && *ab_back == s_ab.bytes,
                     "sampled in_proj_a reconstructs exactly");
            }
          }
        }
      }
    }
  } else {
    fail("authoritative checkpoint is missing");
  }

  if (g_failures != 0) {
    std::cerr << g_failures << " failures\n";
    return 1;
  }
  return 0;
}
