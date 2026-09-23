#include "runtime/arena.hpp"
#include "runtime/model.hpp"
#include "runtime/sizes.hpp"
#include "runtime/view.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <string_view>

using qw38::format::ScratchKind;
using qw38::runtime::ErrorCode;
using qw38::runtime::find_placement;
using qw38::runtime::intervals_overlap;
using qw38::runtime::KernelStage;
using qw38::runtime::kArenaTokenCapacity;
using qw38::runtime::kAttentionWorkspaceBytesPerToken;
using qw38::runtime::kFixedPersistentBytes;
using qw38::runtime::kGdnWorkspaceBytesPerToken;
using qw38::runtime::kKvBytesPerToken;
using qw38::runtime::kLogitsBytesPerToken;
using qw38::runtime::kMaxKvCapacity;
using qw38::runtime::kNormalizedBytesPerToken;
using qw38::runtime::kv_cache_bytes;
using qw38::runtime::LiveInterval;
using qw38::runtime::persistent_state_bytes;
using qw38::runtime::plan_scratch_arena;
using qw38::runtime::plan_v0_arena;
using qw38::runtime::ScratchRequest;
using qw38::runtime::validate_kv_capacity;
using qw38::runtime::v0_scratch_requests;

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

}  // namespace

int main() {
  {
    using qw38::format::GraphBinding;
    using qw38::format::SemanticNodeKind;
    using qw38::format::TensorRecord;
    using qw38::format::TensorRole;
    qw38::format::ArtifactSchema schema{};
    schema.tensors = {
        TensorRecord{.tensor_id = 9, .logical_name = "model.language_model.layers.3.self_attn.q_norm.weight"},
        TensorRecord{.tensor_id = 2, .logical_name = "model.language_model.layers.3.self_attn.k_norm.weight"},
        TensorRecord{.tensor_id = 8, .logical_name = "model.language_model.layers.7.self_attn.q_norm.weight"},
    };
    schema.graph_bindings = {
        GraphBinding{.instance_id = 7, .kind = SemanticNodeKind::GatedAttention,
                     .role = TensorRole::QkNorm, .layer_index = 3, .tensor_id = 9},
        GraphBinding{.instance_id = 7, .kind = SemanticNodeKind::GatedAttention,
                     .role = TensorRole::QkNorm, .layer_index = 3, .tensor_id = 2},
        GraphBinding{.instance_id = 15, .kind = SemanticNodeKind::GatedAttention,
                     .role = TensorRole::QkNorm, .layer_index = 7, .tensor_id = 8},
    };
    auto resolve = [&](std::string_view name, std::uint32_t layer) {
      return qw38::runtime::resolve_semantic_tensor(
          schema, name, SemanticNodeKind::GatedAttention, TensorRole::QkNorm,
          layer);
    };
    auto q3 = resolve(schema.tensors[0].logical_name, 3);
    auto k3 = resolve(schema.tensors[1].logical_name, 3);
    auto q7 = resolve(schema.tensors[2].logical_name, 7);
    expect(q3 && k3 && q7 && *q3 == 9 && *k3 == 2 && *q7 == 8,
           "canonical names select distinct same-shaped and per-layer IDs");
    std::swap(schema.tensors[0], schema.tensors[2]);
    std::swap(schema.graph_bindings[0], schema.graph_bindings[2]);
    expect(resolve("model.language_model.layers.3.self_attn.q_norm.weight", 3) == q3,
           "nonsemantic directory and graph order is ignored");
    auto bad = schema;
    bad.graph_bindings[1].layer_index = 7;
    expect(!qw38::runtime::resolve_semantic_tensor(
               bad, "model.language_model.layers.3.self_attn.k_norm.weight",
               SemanticNodeKind::GatedAttention, TensorRole::QkNorm, 3),
           "wrong-layer graph association rejected");
    bad = schema;
    bad.graph_bindings.push_back(bad.graph_bindings[1]);
    expect(!qw38::runtime::resolve_semantic_tensor(
               bad, "model.language_model.layers.3.self_attn.k_norm.weight",
               SemanticNodeKind::GatedAttention, TensorRole::QkNorm, 3),
           "duplicate binding rejected");
    bad = schema;
    bad.graph_bindings.erase(bad.graph_bindings.begin() + 1);
    expect(!qw38::runtime::resolve_semantic_tensor(
               bad, "model.language_model.layers.3.self_attn.k_norm.weight",
               SemanticNodeKind::GatedAttention, TensorRole::QkNorm, 3),
           "missing binding rejected");
  }
  {
    qw38::runtime::TensorView tensor{};
    tensor.pointer = reinterpret_cast<void*>(0x1000);
    tensor.dtype = qw38::format::ArithmeticDtype::Bf16;
    tensor.rank = 1;
    tensor.extent[0] = 4;
    auto workspace = qw38::runtime::WorkspaceView::from_tensor(tensor);
    expect(workspace && workspace->bytes == 8 && workspace->region_count == 1,
           "valid tensor converts to one bounded workspace region");
    for (std::uint8_t rank : {std::uint8_t{9}, std::uint8_t{255}}) {
      tensor.rank = rank;
      expect(!tensor.extents() && !qw38::runtime::WorkspaceView::from_tensor(tensor),
             "out-of-range tensor rank rejected before extent access");
    }
    tensor.rank = 0;
    expect(!qw38::runtime::WorkspaceView::from_tensor(tensor),
           "zero-rank workspace tensor rejected");
    tensor.rank = 2;
    tensor.extent[0] = std::numeric_limits<std::uint64_t>::max();
    tensor.extent[1] = 2;
    auto elements_overflow = qw38::runtime::WorkspaceView::from_tensor(tensor);
    expect(!elements_overflow && elements_overflow.error().code == ErrorCode::Overflow,
           "workspace element product overflow rejected");
    tensor.rank = 1;
    auto bytes_overflow = qw38::runtime::WorkspaceView::from_tensor(tensor);
    expect(!bytes_overflow && bytes_overflow.error().code == ErrorCode::Overflow,
           "workspace byte product overflow rejected");
    tensor.extent[0] = 0;
    expect(!qw38::runtime::WorkspaceView::from_tensor(tensor),
           "zero workspace extent rejected");
    qw38::runtime::WorkspaceView regions{};
    for (std::uint8_t count : {std::uint8_t{13}, std::uint8_t{255}}) {
      regions.region_count = count;
      expect(!regions.regions(), "out-of-range region count rejected");
    }
  }
  expect(intervals_overlap({KernelStage::Mixer, KernelStage::MixerOutput},
                           {KernelStage::Mixer, KernelStage::Mixer}),
         "mixer intervals overlap");
  expect(!intervals_overlap({KernelStage::Mixer, KernelStage::MixerOutput},
                            {KernelStage::MlpSwiglu, KernelStage::MlpDown}),
         "mixer and swiglu do not overlap");

  auto t0 = persistent_state_bytes(0);
  expect(t0 && *t0 == kFixedPersistentBytes, "T=0 is fixed S+conv");
  auto t1 = persistent_state_bytes(1);
  expect(t1 && *t1 == kFixedPersistentBytes + kKvBytesPerToken, "T=1 state size");
  auto t256 = persistent_state_bytes(256);
  expect(t256 && *t256 == kFixedPersistentBytes + kKvBytesPerToken * 256,
         "T=256 state size");

  std::uint64_t const overflow_t =
      std::numeric_limits<std::uint64_t>::max() / kKvBytesPerToken + 1;
  auto ov = persistent_state_bytes(overflow_t);
  expect(!ov && ov.error().code == ErrorCode::Overflow,
         "capacity multiply overflow");
  auto kv_ov = kv_cache_bytes(overflow_t);
  expect(!kv_ov && kv_ov.error().code == ErrorCode::Overflow, "KV overflow");

  auto reqs1 = v0_scratch_requests(1);
  expect(static_cast<bool>(reqs1), "decode-sized scratch requests");
  auto plan1 = plan_scratch_arena(*reqs1);
  expect(static_cast<bool>(plan1), "plan decode-sized arena");
  if (plan1) {
    auto const* gdn = find_placement(*plan1, ScratchKind::GdnWorkspace);
    auto const* attn = find_placement(*plan1, ScratchKind::AttentionWorkspace);
    auto const* sw = find_placement(*plan1, ScratchKind::MlpSwiglu);
    auto const* logits = find_placement(*plan1, ScratchKind::Logits);
    auto const* norm = find_placement(*plan1, ScratchKind::NormalizedHidden);
    expect(gdn && attn && sw && logits && norm, "all V0 scratch kinds placed");
    if (gdn && attn && sw && logits && norm) {
      expect(gdn->offset == attn->offset, "GDN/attention exclusive-share offset");
      expect(gdn->bytes == kGdnWorkspaceBytesPerToken, "GDN workspace size");
      expect(attn->bytes == kAttentionWorkspaceBytesPerToken,
             "attention workspace size");
      expect(!gdn->dtype && !attn->dtype,
             "mixed workspaces are composite byte arenas");
      expect(sw->offset == gdn->offset, "SwiGLU reuses mixer lifetime");
      expect(logits->offset == gdn->offset, "logits reuse mixer lifetime");
      expect(norm->offset != gdn->offset, "normalized stays live across mixer");
      expect(plan1->total_bytes == kLogitsBytesPerToken + kNormalizedBytesPerToken,
             "packed decode arena size");
    }
  }

  auto plan256 = plan_v0_arena(kArenaTokenCapacity);
  expect(static_cast<bool>(plan256), "T-02 256-token arena");
  if (plan256) {
    auto const* gdn = find_placement(*plan256, ScratchKind::GdnWorkspace);
    auto const* attn = find_placement(*plan256, ScratchKind::AttentionWorkspace);
    auto const* norm = find_placement(*plan256, ScratchKind::NormalizedHidden);
    expect(gdn && attn && norm, "256-token placements");
    if (gdn && attn && norm) {
      expect(gdn->offset == attn->offset, "256-token exclusive mixer share");
      expect(gdn->bytes == kGdnWorkspaceBytesPerToken * kArenaTokenCapacity,
             "256-token GDN workspace");
      expect(norm->bytes == kNormalizedBytesPerToken * kArenaTokenCapacity,
             "256-token normalized");
      expect(plan256->total_bytes ==
                 (kLogitsBytesPerToken + kNormalizedBytesPerToken) *
                     kArenaTokenCapacity,
             "256-token packed arena");
    }
  }

  struct AttentionCapacityCase {
    std::uint64_t capacity;
    std::uint64_t workspace_bytes;
  };
  constexpr std::array kAttentionCapacityCases{
      AttentionCapacityCase{1, kAttentionWorkspaceBytesPerToken},
      AttentionCapacityCase{205824, 19966720},
      AttentionCapacityCase{205825, 19991488},
      AttentionCapacityCase{kMaxKvCapacity, 25415680},
  };
  for (auto const& c : kAttentionCapacityCases) {
    auto plan = plan_v0_arena(kArenaTokenCapacity, c.capacity);
    expect(static_cast<bool>(plan), "capacity-derived attention arena");
    if (!plan) {
      continue;
    }
    auto const* attention = find_placement(*plan, ScratchKind::AttentionWorkspace);
    expect(attention && attention->bytes == c.workspace_bytes,
           "attention partial workspace covers requested KV capacity");
  }
  auto min_capacity = validate_kv_capacity(1);
  auto max_capacity = validate_kv_capacity(kMaxKvCapacity);
  auto zero_capacity = validate_kv_capacity(0);
  auto over_capacity = validate_kv_capacity(kMaxKvCapacity + 1);
  expect(static_cast<bool>(min_capacity), "minimum KV capacity accepted");
  expect(static_cast<bool>(max_capacity), "maximum KV capacity accepted");
  expect(!zero_capacity && zero_capacity.error().code == ErrorCode::InvalidCapacity,
         "zero KV capacity rejected");
  expect(!over_capacity && over_capacity.error().code == ErrorCode::InvalidCapacity,
         "one-past-maximum KV capacity rejected");
  auto over_capacity_plan =
      plan_v0_arena(kArenaTokenCapacity, kMaxKvCapacity + 1);
  expect(!over_capacity_plan &&
             over_capacity_plan.error().code == ErrorCode::InvalidCapacity,
         "arena rejects one-past-maximum attention capacity");
  auto attention_overflow = qw38::runtime::attn_workspace_bytes_for_capacity(
      std::numeric_limits<std::uint64_t>::max());
  expect(!attention_overflow && attention_overflow.error().code == ErrorCode::Overflow,
         "attention workspace arithmetic is checked");

  ScratchRequest a{};
  a.kind = ScratchKind::GdnWorkspace;
  a.bytes = 1024;
  a.live = {{KernelStage::Mixer, KernelStage::Mixer}};
  ScratchRequest b = a;
  b.kind = ScratchKind::NormalizedHidden;
  auto conflict = plan_scratch_arena(std::array{a, b});
  expect(conflict && conflict->total_bytes == 2048, "overlapping live ranges pack sequentially");
  if (conflict && conflict->placements.size() == 2) {
    expect(conflict->placements[0].offset != conflict->placements[1].offset,
           "overlapping ranges get distinct offsets");
  }

  ScratchRequest big{};
  big.kind = ScratchKind::Logits;
  big.bytes = std::numeric_limits<std::uint64_t>::max() - 7;
  big.live = {{KernelStage::LmHead, KernelStage::LmHead}};
  auto align_ov = plan_scratch_arena(std::span(&big, 1), 256);
  expect(!align_ov && align_ov.error().code == ErrorCode::Overflow,
         "arena alignment overflow");

  auto cap0 = v0_scratch_requests(0);
  expect(!cap0 && cap0.error().code == ErrorCode::InvalidArgument,
         "zero arena capacity rejected");

  auto s_off = qw38::runtime::gdn_s_byte_offset(0, 0, 0, 1);
  expect(s_off && *s_off == 4, "S key-contiguous FP32 stride");
  auto s_head = qw38::runtime::gdn_s_byte_offset(0, 1, 0, 0);
  expect(s_head && *s_head == 128ull * 128ull * 4ull, "S value-head stride");
  auto c_off = qw38::runtime::conv_history_byte_offset(0, 1, 0);
  expect(c_off && *c_off == 10240ull * 2ull, "conv tap stride");
  auto kv_off = qw38::runtime::kv_byte_offset(0, 0, 0, 1, 0, 8);
  expect(kv_off && *kv_off == 256ull * 2ull, "KV token stride at capacity 8");
  auto bad = qw38::runtime::gdn_s_byte_offset(48, 0, 0, 0);
  expect(!bad && bad.error().code == ErrorCode::InvalidArgument,
         "S OOB rejected");

  if (g_failures != 0) {
    std::cerr << g_failures << " failures\n";
    return 1;
  }
  std::cout << "runtime_plan ok\n";
  return 0;
}
