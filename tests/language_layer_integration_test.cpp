#include "runtime/language_layer.hpp"

#include "format/format.hpp"
#include "compiler/quantization/reference.hpp"
#include "reference/math.hpp"
#include "cuda/stream.hpp"
#include "runtime/runtime.hpp"
#include "runtime_support.hpp"

#include "cuda/copy.hpp"

#include <array>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace qw38::runtime {
struct LanguageLayerPlanTestAccess {
  static void* mlp_output(LanguageLayerPlan const& plan) {
    return plan.mlp_.next_h.pointer;
  }
  static void fail_mlp_output(LanguageLayerPlan& plan) {
    plan.mlp_.next_h.pointer = nullptr;
  }
  static void restore_mlp_output(LanguageLayerPlan& plan, void* pointer) {
    plan.mlp_.next_h.pointer = pointer;
  }
};
}  // namespace qw38::runtime

namespace {
using namespace qw38;
using namespace qw38::format;
using namespace qw38::runtime;

int failures{};
void check(bool condition, char const* label) {
  if (!condition) {
    std::cerr << "FAIL: " << label << '\n';
    ++failures;
  }
}

bool compare_boundary(std::vector<float> const& got,
                      std::vector<float> const& expected,
                      char const* boundary) {
  if (got.size() == expected.size()) {
    for (std::size_t i = 0; i < got.size(); ++i) {
      if (got[i] != expected[i]) {
        std::cerr << "FAIL: first differing materialized boundary " << boundary
                  << " at element " << i << ": got " << got[i]
                  << ", expected " << expected[i] << '\n';
        ++failures;
        return false;
      }
    }
    return true;
  }
  std::cerr << "FAIL: first differing materialized boundary " << boundary
            << " has extent " << got.size() << ", expected "
            << expected.size() << '\n';
  ++failures;
  return false;
}
bool compare_reference_boundary(std::vector<float> const& got,
                                std::vector<float> const& expected,
                                std::string const& boundary, float abs_tol,
                                float rel_tol) {
  if (got.size() != expected.size()) {
    std::cerr << "FAIL: first differing materialized boundary " << boundary
              << " has extent " << got.size() << ", expected "
              << expected.size() << '\n';
    ++failures;
    return false;
  }
  for (std::size_t i = 0; i < got.size(); ++i) {
    float const limit = abs_tol + rel_tol * std::abs(expected[i]);
    if (!std::isfinite(got[i]) || std::abs(got[i] - expected[i]) > limit) {
      std::cerr << "FAIL: first differing materialized boundary " << boundary
                << " at element " << i << ": got " << got[i]
                << ", reference " << expected[i] << ", tolerance " << limit << '\n';
      ++failures;
      return false;
    }
  }
  return true;
}
bool same_snapshot(SessionSnapshot const& a, SessionSnapshot const& b) {
  return a.gdn_s == b.gdn_s && a.conv_history == b.conv_history &&
         a.kv == b.kv && a.conv_cursor == b.conv_cursor &&
         a.gdn_position == b.gdn_position && a.kv_populated == b.kv_populated;
}
bool equal_bytes(std::vector<std::byte> const& a,
                 std::vector<std::byte> const& b,
                 std::size_t offset, std::size_t bytes) {
  return offset <= a.size() && bytes <= a.size() - offset &&
         offset <= b.size() && bytes <= b.size() - offset &&
         std::equal(a.begin() + static_cast<std::ptrdiff_t>(offset),
                    a.begin() + static_cast<std::ptrdiff_t>(offset + bytes),
                    b.begin() + static_cast<std::ptrdiff_t>(offset));
}

void check_layer_state(SessionSnapshot const& before, SessionSnapshot const& after,
                       std::uint32_t layer, std::uint64_t position) {
  bool const gdn = is_gdn_language_layer(layer);
  auto const index = gdn ? *gdn_state_index(layer) : *attn_state_index(layer);
  auto const s_bytes = before.gdn_s.size() / kGdnLayers;
  auto const conv_bytes = before.conv_history.size() / kConvLayers;
  auto const kv_bytes = before.kv.size() / kAttnLayers;
  bool isolated = true;
  for (std::size_t i = 0; i < kGdnLayers; ++i) {
    bool const changed = gdn && i == index;
    isolated &= equal_bytes(before.gdn_s, after.gdn_s, i * s_bytes, s_bytes) != changed;
    isolated &= equal_bytes(before.conv_history, after.conv_history,
                            i * conv_bytes, conv_bytes) != changed;
    isolated &= after.gdn_position[i] ==
                before.gdn_position[i] + static_cast<std::uint64_t>(changed);
    isolated &= after.conv_cursor[i] ==
                (changed ? (before.conv_cursor[i] + 1u) % 3u
                         : before.conv_cursor[i]);
  }
  for (std::size_t i = 0; i < kAttnLayers; ++i) {
    bool const changed = !gdn && i == index;
    isolated &= equal_bytes(before.kv, after.kv, i * kv_bytes, kv_bytes) != changed;
    isolated &= after.kv_populated[i] ==
                before.kv_populated[i] + static_cast<std::uint64_t>(changed);
  }
  if (!isolated) {
    std::cerr << "FAIL: layer " << layer << " token " << position
              << " changed a non-owning state family/index or failed to advance its own\n";
    ++failures;
  }
}

std::optional<std::vector<std::uint16_t>> read_bf16(
    Artifact const& artifact, std::string const& name) {
  auto const* tensor = artifact.find_tensor(name);
  auto bytes = artifact.payload(name);
  if (!tensor || !bytes) return std::nullopt;
  std::vector<std::byte> row_major;
  if (tensor->shape.rank == 2) {
    auto unpacked = unpack_bf16_dense_tile_v0(
        *bytes, tensor->shape.logical[0], tensor->shape.logical[1]);
    if (!unpacked) return std::nullopt;
    row_major = std::move(*unpacked);
  } else {
    row_major.assign(bytes->begin(), bytes->end());
  }
  std::vector<std::uint16_t> values(row_major.size() / sizeof(std::uint16_t));
  std::memcpy(values.data(), row_major.data(), row_major.size());
  return values;
}

std::optional<std::vector<std::uint16_t>> read_q4(
    Artifact const& artifact, std::string const& name) {
  auto const* tensor = artifact.find_tensor(name);
  auto payload = artifact.payload(name);
  auto scales = artifact.scales(name);
  if (!tensor || !payload || !scales) return std::nullopt;
  auto unpacked = unpack_cuda_v0(tensor->quantizer, tensor->layout,
                                 tensor->shape.logical[0], tensor->shape.logical[1],
                                 *payload, *scales);
  if (!unpacked) return std::nullopt;
  auto bf16 = compiler::dequantize_to_bf16(*unpacked);
  if (!bf16) return std::nullopt;
  return std::move(*bf16);
}

std::optional<std::vector<float>> read_fp32(
    Artifact const& artifact, std::string const& name) {
  auto bytes = artifact.payload(name);
  if (!bytes || bytes->size() % sizeof(float) != 0) return std::nullopt;
  std::vector<float> values(bytes->size() / sizeof(float));
  std::memcpy(values.data(), bytes->data(), bytes->size());
  return values;
}

TensorShape vector_shape(std::uint64_t n) {
  TensorShape shape{};
  shape.rank = 1;
  shape.logical[0] = shape.padded[0] = n;
  return shape;
}
TensorShape matrix_shape(std::uint64_t n, std::uint64_t k) {
  TensorShape shape{};
  shape.rank = 2;
  shape.logical[0] = shape.padded[0] = n;
  shape.logical[1] = shape.padded[1] = k;
  return shape;
}
TensorRecord bf16_vector(std::uint32_t id, std::string name,
                         std::uint64_t n) {
  TensorRecord t{};
  t.tensor_id = id;
  t.logical_name = std::move(name);
  t.shape = vector_shape(n);
  t.storage = StorageClass::Bf16;
  t.layout = PhysicalLayoutId::CudaBf16VectorV0;
  t.mapping.kind = MappingKind::Identity;
  return t;
}
TensorRecord f32_vector(std::uint32_t id, std::string name,
                        std::uint64_t n) {
  auto t = bf16_vector(id, std::move(name), n);
  t.storage = StorageClass::Fp32;
  t.layout = PhysicalLayoutId::CudaFp32VectorV0;
  return t;
}
TensorRecord q4_matrix(std::uint32_t id, std::string name,
                       std::uint64_t n, std::uint64_t k) {
  TensorRecord t{};
  t.tensor_id = id;
  t.logical_name = std::move(name);
  t.shape = matrix_shape(n, k);
  t.storage = StorageClass::Int4Grouped;
  t.quantizer = LogicalQuantizerId::Q4G64V0;
  t.layout = PhysicalLayoutId::CudaQ4G64V0;
  t.mapping = {.kind = MappingKind::DenseTileNK,
               .tile_rows = 8,
               .tile_k = 256,
               .group_size = 64,
               .packed_bytes_per_tile_row = 128};
  return t;
}
TensorRecord bf16_tile(std::uint32_t id, std::string name,
                       std::uint64_t n, std::uint64_t k) {
  TensorRecord t{};
  t.tensor_id = id;
  t.logical_name = std::move(name);
  t.shape = matrix_shape(n, k);
  t.storage = StorageClass::Bf16;
  t.layout = PhysicalLayoutId::CudaBf16DenseTileV0;
  t.mapping = {.kind = MappingKind::DenseTileNK,
               .tile_rows = 8,
               .tile_k = 256,
               .packed_bytes_per_tile_row = 512};
  return t;
}

bool write_fixture(std::filesystem::path const& path,
                   qw38::format::Hash256& manifest_digest,
                   bool wrong_mixer_family = false) {
  constexpr std::uint64_t H = 5120, FF = 17408;
  constexpr std::uint64_t QKV = 10240, Z = 48 * 128;
  constexpr std::uint64_t QG = 24 * 512, AKV = 4 * 256;
  ArtifactSchema schema{};
  schema.compiler = {.ident = "qw38-v0", .major = 0, .minor = 1, .patch = 0};
  schema.source_hash.bytes[0] = 0x10;
  schema.config_hash.bytes[0] = 0x20;
  schema.tokenizer_hash.bytes[0] = 0x30;
  schema.precision = v0_precision_policy();
  schema.scope = SemanticScope::PrimaryLanguage;

  std::uint32_t id = 1;
  auto add = [&](TensorRecord t, SemanticNodeKind kind, TensorRole role,
                 std::uint32_t layer, std::uint32_t instance) {
    auto const tid = t.tensor_id;
    schema.tensors.push_back(std::move(t));
    schema.graph_bindings.push_back(GraphBinding{.instance_id = instance,
        .kind = kind, .role = role, .layer_index = layer, .tensor_id = tid});
  };
  auto gdn = [&](TensorRecord t, TensorRole role, std::uint32_t layer) {
    add(std::move(t), SemanticNodeKind::GatedDeltaNet, role, layer,
        1u + 2u * layer);
  };
  auto attn = [&](TensorRecord t, TensorRole role, std::uint32_t layer) {
    add(std::move(t), SemanticNodeKind::GatedAttention, role, layer,
        1u + 2u * layer);
  };
  auto mlp = [&](TensorRecord t, std::uint32_t layer) {
    auto const role = t.storage == StorageClass::Bf16
                          ? TensorRole::AdditiveNorm
                          : TensorRole::DenseWeight;
    add(std::move(t), SemanticNodeKind::Mlp,
        role, layer, 2u + 2u * layer);
  };

  auto add_gdn_layer = [&](std::uint32_t layer) {
    gdn(bf16_vector(id++, gdn_norm_name(layer), H), TensorRole::AdditiveNorm, layer);
    for (auto [name, n, k] : std::array{
             std::tuple{gdn_qkv_name(layer), QKV, H},
             std::tuple{gdn_z_name(layer), Z, H},
             std::tuple{gdn_out_name(layer), H, Z}}) {
      gdn(q4_matrix(id++, name, n, k), TensorRole::DenseWeight, layer);
    }
    gdn(bf16_tile(id++, gdn_a_name(layer), 48, H), TensorRole::DenseWeight, layer);
    gdn(bf16_tile(id++, gdn_b_name(layer), 48, H), TensorRole::DenseWeight, layer);
    TensorRecord conv{};
    conv.tensor_id = id++;
    conv.logical_name = gdn_conv_name(layer);
    conv.shape.rank = 3;
    conv.shape.logical = {4, 1, QKV, 0, 0, 0};
    conv.shape.padded = conv.shape.logical;
    conv.storage = StorageClass::Bf16;
    conv.layout = PhysicalLayoutId::CudaBf16TapMajorV0;
    conv.mapping.kind = MappingKind::TapMajorConvC1T;
    gdn(std::move(conv), TensorRole::ConvWeight, layer);
    gdn(bf16_vector(id++, gdn_alog_name(layer), 48), TensorRole::TimeParameter, layer);
    gdn(bf16_vector(id++, gdn_dt_name(layer), 48), TensorRole::TimeParameter, layer);
    gdn(bf16_vector(id++, gdn_gated_norm_name(layer), 128),
        TensorRole::GdnGatedNorm, layer);
  };
  auto add_attention_layer = [&](std::uint32_t layer) {
    attn(bf16_vector(id++, attn_norm_name(layer), H), TensorRole::AdditiveNorm, layer);
    attn(q4_matrix(id++, attn_q_name(layer), QG, H), TensorRole::DenseWeight, layer);
    attn(q4_matrix(id++, attn_k_name(layer), AKV, H), TensorRole::DenseWeight, layer);
    attn(q4_matrix(id++, attn_v_name(layer), AKV, H), TensorRole::DenseWeight, layer);
    attn(q4_matrix(id++, attn_o_name(layer), H, 24 * 256), TensorRole::DenseWeight, layer);
    attn(bf16_vector(id++, attn_q_norm_name(layer), 256), TensorRole::QkNorm, layer);
    attn(bf16_vector(id++, attn_k_norm_name(layer), 256), TensorRole::QkNorm, layer);
  };
  add_gdn_layer(0);
  add_gdn_layer(1);
  add_attention_layer(3);
  add_attention_layer(7);
  auto rope = f32_vector(id++, "rope.inv_freq", 32);
  add(std::move(rope), SemanticNodeKind::GatedAttention,
      TensorRole::VectorWeight, 3, 7);
  schema.graph_bindings.push_back(GraphBinding{.instance_id = 15,
      .kind = SemanticNodeKind::GatedAttention, .role = TensorRole::VectorWeight,
      .layer_index = 7, .tensor_id = id - 1});

  auto add_mlp = [&](std::uint32_t layer) {
    mlp(bf16_vector(id++, mlp_norm_name(layer), H), layer);
    mlp(q4_matrix(id++, mlp_gate_name(layer), FF, H), layer);
    mlp(q4_matrix(id++, mlp_up_name(layer), FF, H), layer);
    mlp(q4_matrix(id++, mlp_down_name(layer), H, FF), layer);
  };
  add_mlp(0);
  add_mlp(1);
  add_mlp(3);
  add_mlp(7);
  auto state = language_persistent_schema();
  schema.state.assign(state.begin(), state.end());
  schema.scratch = qw38::runtime::test::language_scratch();
  if (wrong_mixer_family) {
    // A same-shaped dense projection with the wrong mixer graph family must be
    // rejected by the layer binder, before any kernels or state mutations.
    schema.tensors.resize(4);
    schema.graph_bindings.resize(4);
    schema.graph_bindings[3].kind = SemanticNodeKind::GatedAttention;
  }

  auto writer = ArtifactWriter::create(path, schema);
  if (!writer) return false;
  for (auto const& t : schema.tensors) {
    auto payload_n = expected_payload_bytes(t);
    if (!payload_n) return false;
    std::vector<std::byte> payload(*payload_n);
    if (t.storage == StorageClass::Int4Grouped) {
      for (std::size_t i = 0; i < payload.size(); ++i) {
        auto nibble = [](std::size_t index) {
          auto const value = static_cast<unsigned>(index % 14u);
          return value < 7u ? value + 1u : value + 2u;
        };
        auto const lo = nibble(i + t.tensor_id * 3u);
        auto const hi = nibble(i * 7u + t.tensor_id * 5u);
        payload[i] = static_cast<std::byte>(lo | (hi << 4));
      }
    } else if (t.storage == StorageClass::Fp32) {
      for (std::size_t i = 0; i < payload.size() / sizeof(float); ++i) {
        float const value = 0.0025f * static_cast<float>(1u + (i + t.tensor_id) % 13u);
        std::memcpy(payload.data() + i * sizeof(float), &value, sizeof(value));
      }
    } else {
      for (std::size_t i = 0; i < payload.size() / sizeof(std::uint16_t); ++i) {
        auto const signed_step = static_cast<int>((i * 5u + t.tensor_id * 3u) % 13u) - 6;
        float const value = (t.logical_name.find("norm") != std::string::npos)
                                ? static_cast<float>(signed_step) * 0.00390625f
                                : static_cast<float>(signed_step) * 0.015625f;
        auto const bits = qw38::format::fp32_to_bf16_rne(value);
        std::memcpy(payload.data() + i * sizeof(bits), &bits, sizeof(bits));
      }
    }
    if (!writer->write_span(t.logical_name, SpanKind::Payload, payload)) return false;
    auto scales_n = expected_scale_bytes(t);
    if (!scales_n) return false;
    if (*scales_n != 0) {
      std::vector<std::byte> scales(*scales_n);
      bool const is_gdn = t.logical_name.find("linear_attn") != std::string::npos;
      bool const is_mlp = t.logical_name.find("mlp") != std::string::npos;
      auto const scale_base = static_cast<std::uint16_t>(
          is_gdn ? 0x1800u : (is_mlp ? 0x1800u : 0x2800u));
      auto const scale_span = static_cast<unsigned>(0x0100u);
      for (std::size_t i = 0; i < scales.size() / sizeof(std::uint16_t); ++i) {
        // Positive asymmetric FP16 scales distinguish each grouped projection.
        auto const bits = static_cast<std::uint16_t>(scale_base +
            ((i * 11u + t.tensor_id * 17u) % scale_span));
        std::memcpy(scales.data() + i * sizeof(bits), &bits, sizeof(bits));
      }
      if (!writer->write_span(t.logical_name, SpanKind::Scales, scales)) return false;
    }
  }
  auto identity = writer->finalize();
  if (!identity) return false;
  manifest_digest = identity->manifest_digest;
  return true;
}

using TokenBoundaries = std::array<std::array<std::vector<float>, 2>, 4>;

void check_reference_boundaries(Artifact const& artifact,
    std::array<std::vector<float>, 2> const& token_inputs,
    TokenBoundaries const& mixers, TokenBoundaries const& layers) {
  constexpr std::array<std::uint32_t, 4> layer_ids{0, 1, 3, 7};
  for (std::size_t layer_slot = 0; layer_slot < layer_ids.size(); ++layer_slot) {
    auto const layer = layer_ids[layer_slot];
    auto norm = read_bf16(artifact, mlp_norm_name(layer));
    auto gate = read_q4(artifact, mlp_gate_name(layer));
    auto up = read_q4(artifact, mlp_up_name(layer));
    auto down = read_q4(artifact, mlp_down_name(layer));
    if (!norm || !gate || !up || !down) {
      check(false, "decode uploaded MLP tensors for independent BF16 reference");
      return;
    }
    auto compare_mlp = [&](std::vector<float> const& reference_mixer,
                           std::size_t token) {
      auto cpu = reference::decode_mlp_reference(
          reference_mixer, *norm, reference::kDefaultRmsEps,
          *gate, *up, *down);
      if (!cpu) {
        check(false, "independent MLP reference evaluation");
        return;
      }
      compare_reference_boundary(layers[layer_slot][token], cpu->residual,
          "layer " + std::to_string(layer) + " token " + std::to_string(token) +
              " post-MLP residual", reference::tol::kMlpResidualAbs,
          reference::tol::kMlpResidualRel);
    };
    if (is_gdn_language_layer(layer)) {
      auto gamma = read_bf16(artifact, gdn_norm_name(layer));
      auto gated = read_bf16(artifact, gdn_gated_norm_name(layer));
      auto qkv = read_q4(artifact, gdn_qkv_name(layer));
      auto z = read_q4(artifact, gdn_z_name(layer));
      auto a = read_bf16(artifact, gdn_a_name(layer));
      auto b = read_bf16(artifact, gdn_b_name(layer));
      auto out = read_q4(artifact, gdn_out_name(layer));
      auto taps = read_bf16(artifact, gdn_conv_name(layer));
      auto a_log = read_bf16(artifact, gdn_alog_name(layer));
      auto dt = read_bf16(artifact, gdn_dt_name(layer));
      if (!gamma || !gated || !qkv || !z || !a || !b || !out ||
          !taps || !a_log || !dt) {
        check(false, "decode uploaded GDN tensors for independent BF16 reference");
        return;
      }
      std::vector<std::uint16_t> history(
          reference::kConvHistoryTaps * reference::kQkvWidth, 0);
      std::vector<float> state(static_cast<std::size_t>(reference::kGdnValueHeads) *
                               reference::kGdnHeadDim * reference::kGdnHeadDim, 0.0f);
      std::uint32_t cursor = 0;
      for (std::size_t token = 0; token < token_inputs.size(); ++token) {
        auto const& input = layer_slot == 0 ? token_inputs[token]
                                              : layers[layer_slot - 1][token];
        auto cpu = reference::gdn_mixer_reference(input, *gamma, *gated,
            reference::kDefaultRmsEps, *qkv, *z, *a, *b, *out,
            *taps, *a_log, *dt, history, cursor, state);
        if (!cpu) {
          check(false, "independent GDN reference evaluation");
          return;
        }
        history = std::move(cpu->history);
        state = std::move(cpu->s);
        cursor = cpu->cursor;
        compare_reference_boundary(mixers[layer_slot][token], cpu->residual,
            "layer " + std::to_string(layer) + " token " + std::to_string(token) +
                " post-GDN residual", reference::tol::kGdnMixerResidualAbs,
            reference::tol::kGdnMixerResidualRel);
        compare_mlp(cpu->residual, token);
      }
    } else {
      auto gamma = read_bf16(artifact, attn_norm_name(layer));
      auto qg = read_q4(artifact, attn_q_name(layer));
      auto k = read_q4(artifact, attn_k_name(layer));
      auto v = read_q4(artifact, attn_v_name(layer));
      auto out = read_q4(artifact, attn_o_name(layer));
      auto q_norm = read_bf16(artifact, attn_q_norm_name(layer));
      auto k_norm = read_bf16(artifact, attn_k_norm_name(layer));
      auto inv_freq = read_fp32(artifact, "rope.inv_freq");
      if (!gamma || !qg || !k || !v || !out || !q_norm || !k_norm || !inv_freq) {
        check(false, "decode uploaded attention tensors for independent BF16 reference");
        return;
      }
      constexpr std::uint64_t capacity = 3;
      std::vector<std::uint16_t> kv(static_cast<std::size_t>(kAttnLayers) * 2u *
          reference::kKvHeads * capacity * reference::kHeadDim, 0);
      for (std::size_t token = 0; token < token_inputs.size(); ++token) {
        auto const& input = layers[layer_slot - 1][token];
        auto cpu = reference::attn_mixer_reference(input, *gamma,
            reference::kDefaultRmsEps, *qg, *k, *v, *out,
            *q_norm, *k_norm, *inv_freq, kv, *attn_state_index(layer),
            capacity, token, true);
        if (!cpu) {
          check(false, "independent attention reference evaluation");
          return;
        }
        kv = std::move(cpu->kv);
        compare_reference_boundary(mixers[layer_slot][token], cpu->residual,
            "layer " + std::to_string(layer) + " token " + std::to_string(token) +
                " post-attention residual", reference::tol::kAttnMixerResidualAbs,
            reference::tol::kAttnMixerResidualRel);
        compare_mlp(cpu->residual, token);
      }
    }
  }
  if (failures == 0)
    std::cout << "independent BF16 reference: 16 materialized boundaries across "
                 "4 layers and 2 tokens\n";
}

}  // namespace

int main() {
  format::test::ScratchDir dir("qw38-language-layer");
  auto const path = dir.file("hybrid.qw38");
  format::Hash256 manifest_digest{};
  check(write_fixture(path, manifest_digest),
        "write GDN plus attention uploaded-model fixture");
  if (failures) return 1;
  std::cout << "fixture manifest SHA-256=";
  for (auto byte : manifest_digest.bytes) {
    constexpr char hex[] = "0123456789abcdef";
    auto const value = static_cast<unsigned>(byte);
    std::cout << hex[value >> 4] << hex[value & 0x0f];
  }
  std::cout << '\n';
  auto runtime = Runtime::create();
  check(static_cast<bool>(runtime), "create runtime");
  if (!runtime) return 1;
  auto model = runtime->load(path);
  check(static_cast<bool>(model), "upload fixture through Runtime");
  if (!model) std::cerr << "fixture upload error: " << error_message(model.error()) << '\n';
  if (!model) return 1;
  auto const wrong_path = dir.file("wrong-mixer-family.qw38");
  Hash256 wrong_digest{};
  check(write_fixture(wrong_path, wrong_digest, true),
        "write same-shaped wrong-family binding fixture");
  auto wrong_model = runtime->load(wrong_path);
  check(static_cast<bool>(wrong_model), "upload wrong-family binding fixture");
  if (wrong_model) {
    auto wrong_session = runtime->create_session(*wrong_model, 3);
    check(static_cast<bool>(wrong_session), "create wrong-family session");
    if (wrong_session) {
      auto before_bad_bind = wrong_session->save();
      auto wrong_bind = LanguageLayerPlan::bind(*wrong_model, *wrong_session,
                                                 0, runtime->stream());
      auto after_bad_bind = wrong_session->save();
      check(!wrong_bind && wrong_bind.error().code == ErrorCode::MalformedArtifact &&
                before_bad_bind && after_bad_bind &&
                same_snapshot(*before_bad_bind, *after_bad_bind),
            "GDN layer binder rejects attention-family graph binding without state mutation");
      check(static_cast<bool>(wrong_session->shutdown()),
            "shutdown wrong-family session");
    }
  }
  if (failures) return 1;
  auto session = runtime->create_session(*model, 3);
  auto other = runtime->create_session(*model, 3);
  check(session && other, "create isolated sessions");
  if (!session || !other) return 1;

  auto gdn = LanguageLayerPlan::bind(*model, *session, 0, runtime->stream());
  auto gdn1 = LanguageLayerPlan::bind(*model, *session, 1, runtime->stream());
  auto attn = LanguageLayerPlan::bind(*model, *session, 3, runtime->stream());
  auto attn7 = LanguageLayerPlan::bind(*model, *session, 7, runtime->stream());
  auto other_gdn = LanguageLayerPlan::bind(*model, *other, 0, runtime->stream());
  auto other_gdn1 = LanguageLayerPlan::bind(*model, *other, 1, runtime->stream());
  auto other_attn = LanguageLayerPlan::bind(*model, *other, 3, runtime->stream());
  auto other_attn7 = LanguageLayerPlan::bind(*model, *other, 7, runtime->stream());
  check(gdn1 && gdn1->mixer_kind() == LanguageMixerKind::Gdn,
        "layer 1 resolves to a second GDN state index");
  check(gdn && gdn->mixer_kind() == LanguageMixerKind::Gdn,
        "layer 0 resolves to GDN and its graph bindings");
  check(attn && attn->mixer_kind() == LanguageMixerKind::Attention,
        "layer 3 resolves to attention and its graph bindings");
  check(attn7 && attn7->mixer_kind() == LanguageMixerKind::Attention,
        "layer 7 resolves to a second attention state index");
  check(other_gdn && other_gdn1 && other_attn && other_attn7,
        "bind all second-Session layer plans");
  if (!gdn || !gdn1 || !attn || !attn7 || !other_gdn || !other_gdn1 ||
      !other_attn || !other_attn7) return 1;
  auto initial = session->save();
  auto other_initial = other->save();
  check(initial && other_initial, "snapshot both sessions before execution");
  auto alternate_stream = cuda::Stream::create();
  check(static_cast<bool>(alternate_stream), "create alternate same-device stream");
  if (alternate_stream) {
    check(!LanguageLayerPlan::bind(*model, *session, 0, *alternate_stream) &&
              !LanguageLayerPlan::bind(*model, *session, 3, *alternate_stream),
          "both tagged layer kinds reject a non-owning Session stream");
  }
  auto bad = LanguageLayerPlan::bind(*model, *session, 64, runtime->stream());
  check(!bad, "reject out-of-range layer");

  constexpr std::size_t H = 5120;
  std::vector<float> host(H);
  for (std::size_t i = 0; i < H; ++i)
    host[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.125f;
  std::array<std::vector<float>, 2> token_inputs;
  std::array<std::array<std::vector<float>, 2>, 4> mixer_boundaries;
  std::array<std::array<std::vector<float>, 2>, 4> layer_boundaries;
  std::array<LanguageLayerPlan const*, 4> plans{&*gdn, &*gdn1, &*attn, &*attn7};
  token_inputs[0] = host;
  for (std::uint64_t position = 0; position < 2; ++position) {
    if (position != 0) {
      for (std::size_t i = 0; i < H; ++i) host[i] += 0.01f;
      token_inputs[1] = host;
    }
    check(static_cast<bool>(cuda::copy_h2d(session->residual_h().pointer,
          host.data(), H * sizeof(float), runtime->stream())), "upload residual");
    auto previous = host;
    for (std::size_t i = 0; i < plans.size(); ++i) {
      auto before_layer = session->save();
      check(static_cast<bool>(before_layer), "snapshot before individual layer");
      auto result = execute_decode_language_layer(*plans[i], position);
      check(result && result->pointer == session->residual_h().pointer,
            "tagged layer returns Session residual buffer");
      if (!result) break;
      auto& mixer = mixer_boundaries[i][position];
      auto& output = layer_boundaries[i][position];
      mixer.resize(H);
      output.resize(H);
      check(static_cast<bool>(cuda::copy_d2h(mixer.data(),
            session->residual_h_mid().pointer, H * sizeof(float), runtime->stream())),
            "download materialized mixer residual boundary");
      check(static_cast<bool>(cuda::copy_d2h(output.data(), result->pointer,
            H * sizeof(float), runtime->stream())),
            "download materialized MLP residual boundary");
      check(static_cast<bool>(runtime->stream().sync()), "sync layer boundaries");
      auto after_layer = session->save();
      check(static_cast<bool>(after_layer), "snapshot after individual layer");
      if (before_layer && after_layer)
        check_layer_state(*before_layer, *after_layer, plans[i]->layer(), position);
      auto const finite = [](std::vector<float> const& values) {
        return std::all_of(values.begin(), values.end(),
                           [](float x) { return std::isfinite(x); });
      };
      check(finite(mixer) && (i == 3 || mixer != previous),
            "asymmetric fixture changes each mixer residual with finite arithmetic");
      check(finite(output) && output != mixer,
            "post-mixer MLP changes each materialized residual");
      if (output == mixer) {
        std::cerr << "equal MLP boundary at layer-index " << i << " pos " << position
                  << " first " << output[0] << ", " << mixer[0] << '\n';
      }
      previous = output;
    }
  }
  if (failures == 0)
    std::cout << "per-layer state isolation: 8 complete before/after snapshots\n";
  auto artifact = Artifact::open(path);
  check(static_cast<bool>(artifact), "open fixture for independent BF16 reference");
  if (artifact)
    check_reference_boundaries(*artifact, token_inputs,
                               mixer_boundaries, layer_boundaries);

  auto failed_session = runtime->create_session(*model, 3);
  auto control_session = runtime->create_session(*model, 3);
  check(failed_session && control_session,
        "create failed-operation and clean-control sessions");
  if (failed_session && control_session) {
    auto failed_plan = LanguageLayerPlan::bind(*model, *failed_session, 0,
                                                runtime->stream());
    auto control_plan = LanguageLayerPlan::bind(*model, *control_session, 0,
                                                 runtime->stream());
    auto before_failure = failed_session->save();
    check(failed_plan && control_plan && before_failure,
          "bind/snapshot failure and control sessions");
    if (failed_plan && control_plan && before_failure) {
      check(static_cast<bool>(cuda::copy_h2d(failed_session->residual_h().pointer,
            host.data(), H * sizeof(float), runtime->stream())),
            "upload failure-path residual");
      check(static_cast<bool>(cuda::copy_h2d(control_session->residual_h().pointer,
            host.data(), H * sizeof(float), runtime->stream())),
            "upload clean-control residual");
      check(static_cast<bool>(execute_decode_language_layer(*control_plan, 0)),
            "clean-control layer succeeds");
      auto clean = control_session->save();
      void* const valid_output =
          LanguageLayerPlanTestAccess::mlp_output(*failed_plan);
      LanguageLayerPlanTestAccess::fail_mlp_output(*failed_plan);
      auto failed = execute_decode_language_layer(*failed_plan, 0);
      LanguageLayerPlanTestAccess::restore_mlp_output(*failed_plan, valid_output);
      check(!failed && failed.error().code == ErrorCode::InvalidArgument,
            "post-mixer MLP failure is reported after GDN state mutation");
      check(!failed_session->save() &&
                !execute_decode_language_layer(*failed_plan, 0),
            "composite failed operation poisons snapshot and retry");
      check(static_cast<bool>(failed_session->restore(*before_failure)),
            "restore recovers composite failure session");
      check(static_cast<bool>(execute_decode_language_layer(*failed_plan, 0)),
            "composite layer resumes after restore");
      auto recovered = failed_session->save();
      check(clean && recovered &&
                recovered->gdn_s == clean->gdn_s &&
                recovered->conv_history == clean->conv_history &&
                recovered->kv == clean->kv &&
                recovered->conv_cursor == clean->conv_cursor &&
                recovered->gdn_position == clean->gdn_position &&
                recovered->kv_populated == clean->kv_populated,
            "post-mixer failure recovery exactly matches clean control state");
    }
    check(static_cast<bool>(failed_session->shutdown()),
          "shutdown failed-operation session");
    check(static_cast<bool>(control_session->shutdown()),
          "shutdown clean-control session");
  }

  auto snapshot = session->save();
  auto untouched = other->save();
  check(snapshot && untouched, "save full persistent state");
  if (snapshot && untouched && initial && other_initial) {
    check(snapshot->gdn_position[0] == 2 && snapshot->gdn_position[1] == 2 &&
              snapshot->kv_populated[0] == 2 && snapshot->kv_populated[1] == 2 &&
              snapshot->conv_cursor[0] == 2 && snapshot->conv_cursor[1] == 2,
          "distinct GDN and attention state indices each advance by two tokens");
    check(snapshot->gdn_s.size() == kGdnSBytes &&
          snapshot->conv_history.size() == kConvHistoryBytes &&
          snapshot->kv.size() == *kv_cache_bytes(3),
          "snapshot covers full relevant persistent state byte ranges");
    check(snapshot->gdn_s != initial->gdn_s &&
              snapshot->conv_history != initial->conv_history &&
              snapshot->kv != initial->kv,
          "nonzero asymmetric fixture changes complete persistent state byte spans");
    auto const s_layer_bytes = kGdnSBytes / kGdnLayers;
    auto const conv_layer_bytes = kConvHistoryBytes / kConvLayers;
    auto const kv_layer_bytes = snapshot->kv.size() / kAttnLayers;
    check(!equal_bytes(snapshot->gdn_s, initial->gdn_s, 0, s_layer_bytes) &&
              !equal_bytes(snapshot->gdn_s, initial->gdn_s, s_layer_bytes,
                           s_layer_bytes) &&
              equal_bytes(snapshot->gdn_s, initial->gdn_s, 2u * s_layer_bytes,
                          snapshot->gdn_s.size() - 2u * s_layer_bytes),
          "GDN state bytes change at indices 0/1 and later indices remain identical");
    check(!equal_bytes(snapshot->conv_history, initial->conv_history, 0,
                       conv_layer_bytes) &&
              !equal_bytes(snapshot->conv_history, initial->conv_history,
                           conv_layer_bytes, conv_layer_bytes) &&
              equal_bytes(snapshot->conv_history, initial->conv_history,
                          2u * conv_layer_bytes,
                          snapshot->conv_history.size() - 2u * conv_layer_bytes),
          "convolution history bytes change at indices 0/1 only");
    check(!equal_bytes(snapshot->kv, initial->kv, 0, kv_layer_bytes) &&
              !equal_bytes(snapshot->kv, initial->kv, kv_layer_bytes,
                           kv_layer_bytes) &&
              equal_bytes(snapshot->kv, initial->kv, 2u * kv_layer_bytes,
                          snapshot->kv.size() - 2u * kv_layer_bytes),
          "attention KV bytes change at indices 0/1 and later indices remain identical");
    check(same_snapshot(*untouched, *other_initial),
          "second Session remains byte and metadata identical while first executes");
    std::vector<float> other_host(H);
    for (std::size_t i = 0; i < H; ++i) {
      other_host[i] = -host[i] + static_cast<float>(i % 5u) * 0.03125f;
    }
    check(static_cast<bool>(cuda::copy_h2d(other->residual_h().pointer,
          other_host.data(), H * sizeof(float), runtime->stream())),
          "upload distinct second-Session residual");
    check(static_cast<bool>(execute_decode_language_layer(*other_gdn, 0)),
          "execute GDN layer 0 on second Session with distinct input");
    check(static_cast<bool>(execute_decode_language_layer(*other_gdn1, 0)),
          "execute GDN layer 1 on second Session with distinct input");
    check(static_cast<bool>(execute_decode_language_layer(*other_attn, 0)),
          "execute attention layer 3 on second Session with distinct input");
    check(static_cast<bool>(execute_decode_language_layer(*other_attn7, 0)),
          "execute attention layer 7 on second Session with distinct input");
    auto other_after = other->save();
    auto first_after_second = session->save();
    check(other_after && first_after_second &&
              other_after->gdn_position[0] == 1 &&
              other_after->gdn_position[1] == 1 &&
              other_after->kv_populated[0] == 1 &&
              other_after->kv_populated[1] == 1 &&
              other_after->conv_cursor[0] == 1 &&
              other_after->conv_cursor[1] == 1 &&
              other_after->gdn_s != other_initial->gdn_s &&
              other_after->conv_history != other_initial->conv_history &&
              other_after->kv != other_initial->kv,
          "second Session independently commits its own layer state");
    check(first_after_second && same_snapshot(*first_after_second, *snapshot),
          "executing the second Session leaves first Session bytes and metadata unchanged");
    SessionSnapshot restored_continuation;
    auto restored = runtime->create_session(*model, 3);
    check(static_cast<bool>(restored), "create restore continuation session");
    if (restored) {
      check(static_cast<bool>(restored->restore(*snapshot)), "restore snapshot");
      check(static_cast<bool>(cuda::copy_h2d(restored->residual_h().pointer,
            host.data(), H * sizeof(float), runtime->stream())),
            "restore current residual input for continuation");
      auto rg = LanguageLayerPlan::bind(*model, *restored, 0, runtime->stream());
      auto rg1 = LanguageLayerPlan::bind(*model, *restored, 1, runtime->stream());
      auto ra = LanguageLayerPlan::bind(*model, *restored, 3, runtime->stream());
      auto ra7 = LanguageLayerPlan::bind(*model, *restored, 7, runtime->stream());
      check(rg && rg1 && ra && ra7, "bind continuation plans after restore");
      if (rg && rg1 && ra && ra7) {
        check(static_cast<bool>(execute_decode_language_layer(*rg, 2)),
              "restored GDN layer 0 continuation");
        check(static_cast<bool>(execute_decode_language_layer(*rg1, 2)),
              "restored GDN layer 1 continuation");
        check(static_cast<bool>(execute_decode_language_layer(*ra, 2)),
              "restored attention layer 3 continuation");
        check(static_cast<bool>(execute_decode_language_layer(*ra7, 2)),
              "restored attention layer 7 continuation");
        auto restored_snapshot = restored->save();
        check(static_cast<bool>(restored_snapshot), "save restored continuation");
        if (restored_snapshot) restored_continuation = std::move(*restored_snapshot);
      }
    }
    check(static_cast<bool>(session->reset()),
          "reset original Session before exact continuation replay");
    for (std::uint64_t position = 0; position < token_inputs.size(); ++position) {
      check(static_cast<bool>(cuda::copy_h2d(session->residual_h().pointer,
            token_inputs[position].data(), H * sizeof(float), runtime->stream())),
            "upload reset replay residual");
      for (std::size_t i = 0; i < plans.size(); ++i) {
        check(static_cast<bool>(execute_decode_language_layer(*plans[i], position)),
              "execute reset state-index layer replay");
        std::vector<float> replay_mixer(H), replay_layer(H);
        check(static_cast<bool>(cuda::copy_d2h(replay_mixer.data(),
              session->residual_h_mid().pointer, H * sizeof(float), runtime->stream())),
              "download reset replay mixer boundary");
        check(static_cast<bool>(cuda::copy_d2h(replay_layer.data(),
              session->residual_h().pointer, H * sizeof(float), runtime->stream())),
              "download reset replay MLP boundary");
        check(static_cast<bool>(runtime->stream().sync()),
              "sync reset replay boundaries");
        compare_boundary(replay_mixer, mixer_boundaries[i][position],
                         "reset replay mixer residual");
        compare_boundary(replay_layer, layer_boundaries[i][position],
                         "reset replay MLP residual");
      }
    }
    auto reset_replay = session->save();
    check(reset_replay && same_snapshot(*reset_replay, *snapshot),
          "reset and replay exactly reproduces all state bytes and metadata");
    auto moved = std::move(*session);
    check(static_cast<bool>(cuda::copy_h2d(moved.residual_h().pointer,
          host.data(), H * sizeof(float), runtime->stream())),
          "upload matching continuation input after Session move");
    check(static_cast<bool>(execute_decode_language_layer(*gdn, 2)),
          "bound layer plan follows supported Session movement");
    check(static_cast<bool>(execute_decode_language_layer(*gdn1, 2)),
          "second GDN state-index plan follows supported Session movement");
    check(static_cast<bool>(execute_decode_language_layer(*attn, 2)),
          "attention layer plan follows supported Session movement");
    check(static_cast<bool>(execute_decode_language_layer(*attn7, 2)),
          "second attention state-index plan follows supported Session movement");
    auto moved_continuation = moved.save();
    check(moved_continuation &&
              moved_continuation->gdn_s == restored_continuation.gdn_s &&
              moved_continuation->conv_history == restored_continuation.conv_history &&
              moved_continuation->kv == restored_continuation.kv &&
              moved_continuation->conv_cursor == restored_continuation.conv_cursor &&
              moved_continuation->gdn_position == restored_continuation.gdn_position &&
              moved_continuation->kv_populated == restored_continuation.kv_populated,
          "restore continuation equals uninterrupted state byte for byte");
    (void)moved;
  }
  auto session_shutdown = other->shutdown();
  check(static_cast<bool>(session_shutdown), "shutdown second session");
  auto shutdown = runtime->shutdown();
  check(static_cast<bool>(shutdown), "shutdown runtime after composition");
  return failures == 0 ? 0 : 1;
}
