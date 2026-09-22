#include "attention_support.hpp"

#include "cuda/activation.hpp"
#include "cuda/alloc.hpp"
#include "cuda/attention.hpp"
#include "cuda/copy.hpp"
#include "cuda/decode_mmv.hpp"
#include "runtime/runtime.hpp"
#include "runtime_support.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <vector>

using qw38::attn::test::DeviceAttn;
using qw38::attn::test::HostAttn;
using qw38::attn::test::attention_workspace_view;
using qw38::attn::test::bind_views;
using qw38::attn::test::download_vec;
using qw38::attn::test::expect;
using qw38::attn::test::expect_bf16_close;
using qw38::attn::test::fail;
using qw38::attn::test::g_failures;
using qw38::attn::test::kAttnKvWidth;
using qw38::attn::test::kDefaultRmsEps;
using qw38::attn::test::kHeadDim;
using qw38::attn::test::kHidden;
using qw38::attn::test::kKvHeads;
using qw38::attn::test::kQgWidth;
using qw38::attn::test::kQueryHeads;
using qw38::attn::test::kRotaryDim;
using qw38::attn::test::kv_index;
using qw38::attn::test::make_host_q4;
using qw38::attn::test::make_view;
using qw38::attn::test::pattern_h;
using qw38::attn::test::upload_host_attn;
using qw38::attn::test::upload_vec;
using qw38::cuda::DeviceBuffer;
using qw38::cuda::Stream;
using qw38::cuda::launch_attention_prepare;
using qw38::format::ArithmeticDtype;
using qw38::format::PhysicalLayoutId;
using qw38::format::StorageClass;
using qw38::reference::attn_qk_norm_rope;
using qw38::reference::attn_split_qg;
using qw38::reference::kv_head_for_query;
using qw38::reference::qk_rms_norm_1p_gamma;
using qw38::runtime::AttentionPrepBindViews;
using qw38::runtime::AttentionMixerBindViews;
using qw38::runtime::attn_state_index;
using qw38::runtime::bind_attention_prep_plan;
using qw38::runtime::bind_attention_mixer_plan;
using qw38::runtime::bind_attention_workspace;
using qw38::runtime::execute_decode_attention_prep;
using qw38::runtime::is_attention_language_layer;
using qw38::runtime::kAttnBytesG;
using qw38::runtime::kAttnBytesQ;
using qw38::runtime::kAttnOffG;
using qw38::runtime::kAttnOffQ;
using qw38::runtime::kAttnOffQg;
using qw38::runtime::kAttentionWorkspaceBytesPerToken;
using qw38::runtime::kGqaGroup;
using qw38::runtime::kv_byte_offset;

namespace {

void* dummy_ptr(std::uintptr_t v) { return reinterpret_cast<void*>(v); }

AttentionPrepBindViews dummy_ok_views(std::uint64_t* populated,
                                      std::uint64_t capacity) {
  AttentionPrepBindViews v;
  auto* ws = dummy_ptr(0x01000000);
  v.qg = make_view(dummy_ptr(0x10000000), ArithmeticDtype::Bf16,
                   PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped, false,
                   2, kQgWidth, kHidden);
  v.k = make_view(dummy_ptr(0x20000000), ArithmeticDtype::Bf16,
                  PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped, false, 2,
                  kAttnKvWidth, kHidden);
  v.v = make_view(dummy_ptr(0x30000000), ArithmeticDtype::Bf16,
                  PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped, false, 2,
                  kAttnKvWidth, kHidden);
  std::uint64_t const qg_scales =
      static_cast<std::uint64_t>(kQgWidth) * (kHidden / 64u);
  std::uint64_t const kv_scales =
      static_cast<std::uint64_t>(kAttnKvWidth) * (kHidden / 64u);
  v.qg_scales = make_view(dummy_ptr(0x40000000), ArithmeticDtype::Fp16,
                          PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped,
                          false, 1, qg_scales);
  v.k_scales = make_view(dummy_ptr(0x50000000), ArithmeticDtype::Fp16,
                         PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped,
                         false, 1, kv_scales);
  v.v_scales = make_view(dummy_ptr(0x60000000), ArithmeticDtype::Fp16,
                         PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped,
                         false, 1, kv_scales);
  v.gamma = make_view(dummy_ptr(0x70000000), ArithmeticDtype::Bf16,
                      PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16, false,
                      1, kHidden);
  v.gamma_q = make_view(dummy_ptr(0x71000000), ArithmeticDtype::Bf16,
                        PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16,
                        false, 1, kHeadDim);
  v.gamma_k = make_view(dummy_ptr(0x72000000), ArithmeticDtype::Bf16,
                        PhysicalLayoutId::CudaBf16VectorV0, StorageClass::Bf16,
                        false, 1, kHeadDim);
  v.inv_freq = make_view(dummy_ptr(0x73000000), ArithmeticDtype::Fp32,
                         PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32,
                         false, 1, 32);
  v.residual = make_view(dummy_ptr(0x74000000), ArithmeticDtype::Fp32,
                         PhysicalLayoutId::CudaFp32VectorV0, StorageClass::Fp32, true,
                         2, 1, kHidden);
  v.normalized = make_view(dummy_ptr(0x75000000), ArithmeticDtype::Bf16,
                           PhysicalLayoutId::CudaBf16RowMajorV0, StorageClass::Bf16,
                           true, 1, kHidden);
  v.workspace = attention_workspace_view(ws);
  v.kv = make_view(dummy_ptr(0x76000000), ArithmeticDtype::Bf16,
                   PhysicalLayoutId::CudaBf16KvCacheV0, StorageClass::Bf16, true, 5,
                   16, 2);
  v.kv.extent = {16, 2, kKvHeads, capacity, kHeadDim};
  auto slot = qw38::runtime::KvPopulatedSlot::bind(populated, capacity);
  if (slot) {
    v.populated = *slot;
  }
  v.kv_capacity = capacity;
  v.language_layer = 3;
  return v;
}

void test_bind_errors(Stream const& stream) {
  std::uint64_t populated = 0;
  auto views = dummy_ok_views(&populated, 8);
  auto ok = bind_attention_prep_plan(views, stream);
  expect(static_cast<bool>(ok), "valid dummy bind");
  if (ok) {
    expect(ok->scratch.qg.extent[0] == kQueryHeads &&
               ok->scratch.qg.extent[1] == 2u * kHeadDim,
           "qg is [24,512]");
    expect(ok->scratch.q.extent[0] == kQueryHeads &&
               ok->scratch.q.extent[1] == kHeadDim,
           "prepared Q is [24,256]");
    expect(ok->scratch.g.extent[0] == kQueryHeads &&
               ok->scratch.g.extent[1] == kHeadDim,
           "g scratch is aligned per query head");
    expect(ok->scratch.k.extent[0] == kKvHeads && ok->scratch.v.extent[0] == kKvHeads,
           "projected K/V are 4 heads");
    expect(static_cast<std::byte*>(ok->scratch.q.pointer) -
                   static_cast<std::byte*>(ok->scratch.qg.pointer) ==
               static_cast<std::ptrdiff_t>(kAttnOffQ),
           "Q scratch offset");
    expect(static_cast<std::byte*>(ok->scratch.g.pointer) -
                   static_cast<std::byte*>(ok->scratch.qg.pointer) ==
               static_cast<std::ptrdiff_t>(kAttnOffG),
           "g scratch offset");
    expect(ok->scratch.q.extent[0] * ok->scratch.q.extent[1] * 2u == kAttnBytesQ,
           "Q bytes");
    expect(ok->scratch.g.extent[0] * ok->scratch.g.extent[1] * 2u == kAttnBytesG,
           "g bytes");
    expect(ok->attn_layer == 0, "language 3 → attention 0");
  }

  Stream empty;
  auto bad_stream = bind_attention_prep_plan(views, empty);
  expect(!bad_stream &&
             bad_stream.error().code == qw38::runtime::ErrorCode::InvalidArgument,
         "empty stream rejects");

  auto gdn = views;
  gdn.language_layer = 0;
  auto bad_layer = bind_attention_prep_plan(gdn, stream);
  expect(!bad_layer, "GDN layer rejects");

  auto idx = attn_state_index(7);
  expect(static_cast<bool>(idx) && *idx == 1, "language 7 → attention 1");
  expect(is_attention_language_layer(3) && !is_attention_language_layer(4),
         "full-attention layer predicate");
  expect(qw38::runtime::kv_head_for_query(0) == 0 &&
             qw38::runtime::kv_head_for_query(5) == 0 &&
             qw38::runtime::kv_head_for_query(6) == 1 &&
             qw38::runtime::kv_head_for_query(23) == 3,
         "six query heads map to each KV head");
  expect(kQueryHeads / kKvHeads == kGqaGroup, "GQA group size");

  auto no_pop = views;
  no_pop.populated = {};
  expect(!bind_attention_prep_plan(no_pop, stream), "missing populated pointer");

  auto cap0 = views;
  cap0.kv_capacity = 0;
  auto bad_cap = bind_attention_prep_plan(cap0, stream);
  expect(!bad_cap && bad_cap.error().code == qw38::runtime::ErrorCode::InvalidCapacity,
         "zero capacity is typed");
  auto unsupported_cap = views;
  unsupported_cap.kv_capacity = qw38::runtime::kMaxKvCapacity + 1;
  auto bad_unsupported = bind_attention_prep_plan(unsupported_cap, stream);
  expect(!bad_unsupported &&
             bad_unsupported.error().code ==
                 qw38::runtime::ErrorCode::InvalidCapacity,
         "unsupported capacity is typed");

  populated = 9;
  auto bad_pop = qw38::runtime::KvPopulatedSlot::bind(&populated, 8);
  expect(!bad_pop &&
             bad_pop.error().code ==
                 qw38::runtime::ErrorCode::InvalidPopulatedLength,
         "populated > capacity is typed");
  populated = 0;
  expect(ok && !ok->populated.commit_append(1) && populated == 0,
         "out-of-order append commit cannot mutate populated length");

  auto ws = bind_attention_workspace(views.workspace);
  expect(static_cast<bool>(ws), "workspace bind");
  expect(kAttnOffQg == 0, "qg is first workspace alias");

  auto rejects = [&](AttentionPrepBindViews const& bad, std::string const& label) {
    expect(!bind_attention_prep_plan(bad, stream), label);
  };

  auto bad_payload_pointer = views;
  bad_payload_pointer.qg.pointer = nullptr;
  rejects(bad_payload_pointer, "payload null pointer rejects");
  auto bad_payload_space = views;
  bad_payload_space.qg.space = qw38::runtime::MemorySpace::Host;
  rejects(bad_payload_space, "payload host space rejects");
  auto bad_payload_dtype = views;
  bad_payload_dtype.qg.dtype = ArithmeticDtype::Fp32;
  rejects(bad_payload_dtype, "payload arithmetic dtype rejects");
  auto bad_payload_layout = views;
  bad_payload_layout.qg.layout = PhysicalLayoutId::CudaBf16RowMajorV0;
  rejects(bad_payload_layout, "payload physical layout rejects");
  auto bad_payload_storage = views;
  bad_payload_storage.qg.storage = StorageClass::Bf16;
  rejects(bad_payload_storage, "payload storage rejects");
  auto bad_payload_rank = views;
  bad_payload_rank.qg.rank = 1;
  rejects(bad_payload_rank, "payload rank rejects");
  auto bad_payload_extent = views;
  bad_payload_extent.qg.extent[1] = kHidden + 1;
  rejects(bad_payload_extent, "payload extent rejects");
  auto excessive_payload_rank = views;
  excessive_payload_rank.qg.rank = qw38::format::kMaxRank + 1;
  rejects(excessive_payload_rank, "payload rank above maximum rejects");

  auto bad_scale_pointer = views;
  bad_scale_pointer.qg_scales.pointer = nullptr;
  rejects(bad_scale_pointer, "Q4 scale null pointer rejects");
  auto bad_scale_space = views;
  bad_scale_space.qg_scales.space = qw38::runtime::MemorySpace::Host;
  rejects(bad_scale_space, "Q4 scale host space rejects");
  auto bad_scale_dtype = views;
  bad_scale_dtype.qg_scales.dtype = ArithmeticDtype::Bf16;
  rejects(bad_scale_dtype, "Q4 scales require FP16 typed view");
  auto bad_scale_layout = views;
  bad_scale_layout.k_scales.layout = PhysicalLayoutId::CudaBf16VectorV0;
  rejects(bad_scale_layout, "Q4 scales require matching physical layout");
  auto bad_scale_storage = views;
  bad_scale_storage.v_scales.storage = StorageClass::Bf16;
  rejects(bad_scale_storage, "Q4 scales require grouped storage");
  auto bad_scale_rank = views;
  bad_scale_rank.qg_scales.rank = 2;
  rejects(bad_scale_rank, "Q4 scale rank rejects");
  auto bad_scale_extent = views;
  ++bad_scale_extent.qg_scales.extent[0];
  rejects(bad_scale_extent, "Q4 scale extent rejects");

  auto bad_gamma_extent = views;
  --bad_gamma_extent.gamma.extent[0];
  rejects(bad_gamma_extent, "gamma exact extent rejects");
  auto bad_gamma_q_rank = views;
  bad_gamma_q_rank.gamma_q.rank = 2;
  bad_gamma_q_rank.gamma_q.extent = {1, kHeadDim};
  rejects(bad_gamma_q_rank, "gamma_q flattened-equivalent rank rejects");
  auto bad_gamma_k_layout = views;
  bad_gamma_k_layout.gamma_k.layout = PhysicalLayoutId::CudaBf16RowMajorV0;
  rejects(bad_gamma_k_layout, "gamma_k layout rejects");
  auto bad_inv_storage = views;
  bad_inv_storage.inv_freq.storage = StorageClass::Bf16;
  rejects(bad_inv_storage, "inv_freq storage rejects");
  auto bad_residual_dtype = views;
  bad_residual_dtype.residual.dtype = ArithmeticDtype::Bf16;
  rejects(bad_residual_dtype, "residual dtype rejects");
  auto bad_normalized_extent = views;
  ++bad_normalized_extent.normalized.extent[0];
  rejects(bad_normalized_extent, "normalized extent rejects");

  auto bad_workspace_kind = views;
  bad_workspace_kind.workspace.kind = qw38::format::ScratchKind::GdnWorkspace;
  rejects(bad_workspace_kind, "workspace kind rejects");
  auto bad_workspace_space = views;
  bad_workspace_space.workspace.space = qw38::runtime::MemorySpace::Host;
  rejects(bad_workspace_space, "workspace space rejects");
  auto bad_workspace = views;
  bad_workspace.workspace.bytes = 1;
  rejects(bad_workspace, "workspace exact byte extent rejects");
  auto bad_workspace_count = views;
  bad_workspace_count.workspace.region_count = 5;
  rejects(bad_workspace_count, "workspace region count rejects");
  auto bad_workspace_dtype = views;
  bad_workspace_dtype.workspace.region[0].tensor.dtype = ArithmeticDtype::Fp32;
  rejects(bad_workspace_dtype, "workspace region dtype rejects");
  auto bad_workspace_layout = views;
  bad_workspace_layout.workspace.region[0].tensor.layout =
      PhysicalLayoutId::CudaBf16VectorV0;
  rejects(bad_workspace_layout, "workspace region layout rejects");
  auto bad_workspace_storage = views;
  bad_workspace_storage.workspace.region[0].tensor.storage = StorageClass::Fp32;
  rejects(bad_workspace_storage, "workspace region storage rejects");
  auto bad_workspace_rank = views;
  bad_workspace_rank.workspace.region[0].tensor.rank = 1;
  rejects(bad_workspace_rank, "workspace region rank rejects");
  auto bad_workspace_extent = views;
  ++bad_workspace_extent.workspace.region[0].tensor.extent[0];
  rejects(bad_workspace_extent, "workspace region extent rejects");
  auto bad_workspace_offset = views;
  ++bad_workspace_offset.workspace.region[0].offset;
  rejects(bad_workspace_offset, "workspace region offset rejects");
  auto bad_workspace_region_bytes = views;
  ++bad_workspace_region_bytes.workspace.region[0].bytes;
  rejects(bad_workspace_region_bytes, "workspace region byte count rejects");
  auto bad_workspace_stride = views;
  ++bad_workspace_stride.workspace.region[0].stride_bytes;
  rejects(bad_workspace_stride, "workspace region stride rejects");
  auto bad_workspace_repetitions = views;
  bad_workspace_repetitions.workspace.region[0].repetitions = 2;
  rejects(bad_workspace_repetitions, "workspace region repetitions reject");
  auto bad_workspace_region_pointer = views;
  bad_workspace_region_pointer.workspace.region[0].tensor.pointer =
      dummy_ptr(0x01000010);
  rejects(bad_workspace_region_pointer, "workspace region pointer rejects");

  auto bad_kv_layout = views;
  bad_kv_layout.kv.layout = PhysicalLayoutId::CudaBf16RowMajorV0;
  rejects(bad_kv_layout, "KV requires cache physical layout");
  auto bad_kv_pointer = views;
  bad_kv_pointer.kv.pointer = nullptr;
  rejects(bad_kv_pointer, "KV null pointer rejects");
  auto bad_kv_space = views;
  bad_kv_space.kv.space = qw38::runtime::MemorySpace::Host;
  rejects(bad_kv_space, "KV host space rejects");
  auto bad_kv_dtype = views;
  bad_kv_dtype.kv.dtype = ArithmeticDtype::Fp32;
  rejects(bad_kv_dtype, "KV arithmetic dtype rejects");
  auto bad_kv_storage = views;
  bad_kv_storage.kv.storage = StorageClass::Fp32;
  rejects(bad_kv_storage, "KV storage rejects");
  auto bad_kv_rank = views;
  bad_kv_rank.kv.rank = 1;
  rejects(bad_kv_rank, "KV rank rejects");
  auto bad_kv_extent = views;
  ++bad_kv_extent.kv.extent[2];
  rejects(bad_kv_extent, "KV exact geometry rejects");
  auto excessive_kv_rank = views;
  excessive_kv_rank.kv.rank = qw38::format::kMaxRank + 1;
  rejects(excessive_kv_rank, "KV rank above maximum rejects");
  auto element_overflow = views;
  element_overflow.kv.extent[0] = std::numeric_limits<std::uint64_t>::max();
  element_overflow.kv.extent[1] = 2;
  auto element_overflow_result = bind_attention_prep_plan(element_overflow, stream);
  expect(!element_overflow_result &&
             element_overflow_result.error().code ==
                 qw38::runtime::ErrorCode::Overflow,
         "overflowing KV element product rejects");
  auto byte_overflow = views;
  byte_overflow.normalized.extent[0] =
      std::numeric_limits<std::uint64_t>::max();
  auto byte_overflow_result = bind_attention_prep_plan(byte_overflow, stream);
  expect(!byte_overflow_result &&
             byte_overflow_result.error().code ==
                 qw38::runtime::ErrorCode::Overflow,
         "overflowing view byte product rejects");
  auto misaligned = views;
  misaligned.residual.pointer = dummy_ptr(0x74000002);
  expect(!bind_attention_prep_plan(misaligned, stream),
         "misaligned residual rejects");
  auto misaligned_workspace = views;
  misaligned_workspace.workspace.pointer =
      static_cast<std::byte*>(dummy_ptr(0x01000004));
  expect(!bind_attention_prep_plan(misaligned_workspace, stream),
         "misaligned workspace rejects");
  auto overlap_normalized = views;
  overlap_normalized.normalized.pointer = dummy_ptr(0x74000004);
  expect(!bind_attention_prep_plan(overlap_normalized, stream),
         "residual and normalized offset overlap rejects");
  auto overlap_cache = views;
  overlap_cache.kv.pointer = dummy_ptr(0x01000010);
  expect(!bind_attention_prep_plan(overlap_cache, stream),
         "workspace and cache offset overlap rejects");
  auto overlap_projection_weight = views;
  overlap_projection_weight.qg.pointer = dummy_ptr(0x75000010);
  expect(!bind_attention_prep_plan(overlap_projection_weight, stream),
         "projection weight and normalized input offset overlap rejects");

  AttentionMixerBindViews mixer;
  mixer.prep = views;
  mixer.out = make_view(dummy_ptr(0x80000000), ArithmeticDtype::Bf16,
                        PhysicalLayoutId::CudaQ4G64V0,
                        StorageClass::Int4Grouped, false, 2, kHidden,
                        kQueryHeads * kHeadDim);
  mixer.out_scales = make_view(
      dummy_ptr(0x90000000), ArithmeticDtype::Fp16,
      PhysicalLayoutId::CudaQ4G64V0, StorageClass::Int4Grouped, false, 1,
      static_cast<std::uint64_t>(kHidden) * (kQueryHeads * kHeadDim / 64u));
  mixer.residual_out = make_view(dummy_ptr(0xA0000000), ArithmeticDtype::Fp32,
                                 PhysicalLayoutId::CudaFp32VectorV0,
                                 StorageClass::Fp32, true, 2, 1, kHidden);
  expect(static_cast<bool>(bind_attention_mixer_plan(mixer, stream)),
         "mixer permits only its internal Q to Y reuse");
  mixer.residual_out.pointer = dummy_ptr(0x74000004);
  expect(!bind_attention_mixer_plan(mixer, stream),
         "residual and output offset overlap rejects");
}

void test_cuda_prepare_alias_errors(Stream const& stream) {
  constexpr std::uint64_t kCapacity = 1;
  auto arena = DeviceBuffer::allocate(160u * 1024u);
  if (!arena) {
    fail("CUDA preparation alias arena allocation");
    return;
  }
  auto* base = static_cast<std::byte*>(arena->data());
  auto* qg = reinterpret_cast<std::uint16_t*>(base);
  auto* k_raw = reinterpret_cast<std::uint16_t*>(base + 32768u);
  auto* v_raw = reinterpret_cast<std::uint16_t*>(base + 36864u);
  auto* gamma_q = reinterpret_cast<std::uint16_t*>(base + 40960u);
  auto* gamma_k = reinterpret_cast<std::uint16_t*>(base + 41984u);
  auto* inv_freq = reinterpret_cast<float*>(base + 43008u);
  auto* q_out = reinterpret_cast<std::uint16_t*>(base + 49152u);
  auto* g_out = reinterpret_cast<std::uint16_t*>(base + 65536u);
  auto* kv = reinterpret_cast<std::uint16_t*>(base + 81920u);

  auto launch = [&](std::uint16_t const* k_arg, std::uint16_t* q_arg,
                    std::uint16_t* g_arg) {
    return launch_attention_prepare(
        qg, k_arg, v_raw, gamma_q, gamma_k, inv_freq, kDefaultRmsEps, 0,
        q_arg, g_arg, kv, 0, kCapacity, 0, stream);
  };

  auto projection_overlap = launch(k_raw, k_raw + 1, g_out);
  expect(!projection_overlap &&
             projection_overlap.error().code ==
                 qw38::cuda::ErrorCode::InvalidArgument,
         "CUDA prep rejects offset-overlapping projection input/output");

  auto cache_overlap = launch(k_raw, q_out, kv + 1);
  expect(!cache_overlap &&
             cache_overlap.error().code ==
                 qw38::cuda::ErrorCode::InvalidArgument,
         "CUDA prep rejects offset-overlapping prepared output/cache");

  auto input_cache_overlap = launch(kv + 1, q_out, g_out);
  expect(!input_cache_overlap &&
             input_cache_overlap.error().code ==
                 qw38::cuda::ErrorCode::InvalidArgument,
         "CUDA prep rejects offset-overlapping projection input/cache");
}

void test_qg_ordering_and_suffix(Stream const& stream) {
  std::vector<std::uint16_t> qg(kQgWidth);
  std::vector<std::uint16_t> k_raw(kAttnKvWidth);
  std::vector<std::uint16_t> v_raw(kAttnKvWidth);
  for (std::uint32_t h = 0; h < kQueryHeads; ++h) {
    for (std::uint32_t i = 0; i < kHeadDim; ++i) {
      qg[h * 2u * kHeadDim + i] =
          qw38::format::fp32_to_bf16_rne(0.02f * static_cast<float>(h + 1) +
                                         0.001f * static_cast<float>(i));
      qg[h * 2u * kHeadDim + kHeadDim + i] =
          qw38::format::fp32_to_bf16_rne(1.0f + 0.01f * static_cast<float>(h) +
                                         0.002f * static_cast<float>(i));
    }
  }
  for (std::uint32_t h = 0; h < kKvHeads; ++h) {
    for (std::uint32_t i = 0; i < kHeadDim; ++i) {
      k_raw[h * kHeadDim + i] =
          qw38::format::fp32_to_bf16_rne(0.03f * static_cast<float>(h + 2) +
                                         0.001f * static_cast<float>(i));
      v_raw[h * kHeadDim + i] =
          qw38::format::fp32_to_bf16_rne(0.5f + 0.01f * static_cast<float>(h * kHeadDim + i));
    }
  }
  auto gamma_q = pattern_h(kHeadDim, 0.1f);
  auto gamma_k = pattern_h(kHeadDim, -0.2f);
  auto inv = qw38::reference::rope_inv_freq();
  std::vector<std::uint16_t> q_raw(kQueryHeads * kHeadDim);
  std::vector<std::uint16_t> g_cpu(kQueryHeads * kHeadDim);
  expect(static_cast<bool>(attn_split_qg(qg, q_raw, g_cpu)), "cpu q/g split");
  for (std::uint32_t h = 0; h < kQueryHeads; ++h) {
    expect(q_raw[h * kHeadDim] == qg[h * 2u * kHeadDim], "q is first 256 of head");
    expect(g_cpu[h * kHeadDim] == qg[h * 2u * kHeadDim + kHeadDim],
           "g is second 256 of head");
  }

  std::vector<std::uint16_t> q_cpu(kQueryHeads * kHeadDim);
  std::vector<std::uint16_t> k_cpu(kAttnKvWidth);
  for (std::uint32_t h = 0; h < kQueryHeads; ++h) {
    expect(static_cast<bool>(attn_qk_norm_rope(
               std::span<std::uint16_t const>(q_raw.data() + h * kHeadDim, kHeadDim),
               gamma_q, inv, 1, kDefaultRmsEps,
               std::span<std::uint16_t>(q_cpu.data() + h * kHeadDim, kHeadDim))),
           "cpu Q prep");
  }
  for (std::uint32_t h = 0; h < kKvHeads; ++h) {
    expect(static_cast<bool>(attn_qk_norm_rope(
               std::span<std::uint16_t const>(k_raw.data() + h * kHeadDim, kHeadDim),
               gamma_k, inv, 1, kDefaultRmsEps,
               std::span<std::uint16_t>(k_cpu.data() + h * kHeadDim, kHeadDim))),
           "cpu K prep");
  }

  auto d_qg = upload_vec(qg, stream);
  auto d_k = upload_vec(k_raw, stream);
  auto d_v = upload_vec(v_raw, stream);
  auto d_gq = upload_vec(gamma_q, stream);
  auto d_gk = upload_vec(gamma_k, stream);
  std::vector<float> inv_h(inv.begin(), inv.end());
  auto d_inv = upload_vec(inv_h, stream);
  auto d_q = DeviceBuffer::allocate(q_cpu.size() * 2);
  auto d_g = DeviceBuffer::allocate(g_cpu.size() * 2);
  constexpr std::uint64_t kCap = 4;
  auto d_kv = DeviceBuffer::allocate(static_cast<std::uint64_t>(16) * 2u * kKvHeads *
                                     kCap * kHeadDim * 2u);
  if (!d_qg || !d_k || !d_v || !d_gq || !d_gk || !d_inv || !d_q || !d_g || !d_kv) {
    fail("prep isolation alloc");
    return;
  }
  expect(static_cast<bool>(qw38::cuda::zero(*d_kv, stream)), "zero isolated kv");
  expect(static_cast<bool>(launch_attention_prepare(
             static_cast<std::uint16_t const*>(d_qg->data()),
             static_cast<std::uint16_t const*>(d_k->data()),
             static_cast<std::uint16_t const*>(d_v->data()),
             static_cast<std::uint16_t const*>(d_gq->data()),
             static_cast<std::uint16_t const*>(d_gk->data()),
             static_cast<float const*>(d_inv->data()), kDefaultRmsEps, 1,
             static_cast<std::uint16_t*>(d_q->data()),
             static_cast<std::uint16_t*>(d_g->data()),
             static_cast<std::uint16_t*>(d_kv->data()), 0, kCap, 1, stream)),
         "launch isolated prep");
  auto got_q = download_vec<std::uint16_t>(*d_q, q_cpu.size(), stream);
  auto got_g = download_vec<std::uint16_t>(*d_g, g_cpu.size(), stream);
  auto got_kv = download_vec<std::uint16_t>(
      *d_kv, static_cast<std::size_t>(16) * 2u * kKvHeads * kCap * kHeadDim, stream);
  expect(static_cast<bool>(got_q) && static_cast<bool>(got_g) &&
             static_cast<bool>(got_kv),
         "download isolated prep");
  if (!got_q || !got_g || !got_kv) {
    return;
  }
  expect_bf16_close(*got_g, g_cpu, "g scratch is the unnormalized second half", 0.0f,
                    0.0f);
  expect_bf16_close(*got_q, q_cpu, "prepared Q vs fused reference",
                    qw38::reference::tol::kRmsBf16Abs, 0.0f);

  for (std::uint32_t h = 0; h < kQueryHeads; ++h) {
    std::vector<float> qf(kHeadDim);
    for (std::uint32_t i = 0; i < kHeadDim; ++i) {
      qf[i] = qw38::format::bf16_to_fp32(q_raw[h * kHeadDim + i]);
    }
    std::vector<std::uint16_t> rms_only(kHeadDim);
    expect(static_cast<bool>(qk_rms_norm_1p_gamma(qf, gamma_q, kDefaultRmsEps, rms_only)),
           "cpu RMS-only Q");
    for (std::uint32_t i = kRotaryDim; i < kHeadDim; ++i) {
      float const got = qw38::format::bf16_to_fp32((*got_q)[h * kHeadDim + i]);
      float const ref = qw38::format::bf16_to_fp32(rms_only[i]);
      float const d = got > ref ? got - ref : ref - got;
      if (d > qw38::reference::tol::kRmsBf16Abs) {
        fail("RoPE leaves suffix 192 unchanged after QK norm");
        return;
      }
    }
  }

  for (std::uint32_t h = 0; h < kKvHeads; ++h) {
    std::vector<std::uint16_t> k_got(kHeadDim);
    std::vector<std::uint16_t> v_got(kHeadDim);
    for (std::uint32_t i = 0; i < kHeadDim; ++i) {
      k_got[i] = (*got_kv)[kv_index(0, 0, h, 1, i, kCap)];
      v_got[i] = (*got_kv)[kv_index(0, 1, h, 1, i, kCap)];
    }
    expect_bf16_close(k_got,
                      std::span<std::uint16_t const>(k_cpu.data() + h * kHeadDim,
                                                     kHeadDim),
                      "one K copy per KV head", qw38::reference::tol::kRmsBf16Abs,
                      0.0f);
    expect_bf16_close(v_got,
                      std::span<std::uint16_t const>(v_raw.data() + h * kHeadDim,
                                                     kHeadDim),
                      "V stored once without RoPE", 0.0f, 0.0f);
    auto const other = kv_index(0, 0, h, 0, 0, kCap);
    expect((*got_kv)[other] == 0, "unwritten token stays zero");
  }
}

void test_capacity_and_mismatch(Stream const& stream) {
  HostAttn host;
  if (!make_host_q4(host, 3)) {
    return;
  }
  DeviceAttn dev;
  if (!upload_host_attn(host, dev, stream, 2)) {
    return;
  }
  auto views = bind_views(dev, 3);
  auto plan = bind_attention_prep_plan(views, stream);
  expect(static_cast<bool>(plan), "bind capacity fixture");
  if (!plan) {
    return;
  }
  auto mismatch = execute_decode_attention_prep(*plan, 1);
  expect(!mismatch &&
             mismatch.error().code == qw38::runtime::ErrorCode::InvalidPopulatedLength,
         "position mismatch is typed");
  expect(dev.populated == 0, "mismatch does not advance populated length");

  qw38::cuda::testing::fail_next_stream_sync();
  auto deferred = execute_decode_attention_prep(*plan, 0);
  expect(!deferred && deferred.error().code == qw38::runtime::ErrorCode::Cuda,
         "injected deferred attention append failure is reported");
  expect(dev.populated == 0,
         "failed attention append does not commit populated length");

  expect(static_cast<bool>(execute_decode_attention_prep(*plan, 0)), "append 0");
  expect(dev.populated == 1, "populated is 1 after first success");
  expect(static_cast<bool>(execute_decode_attention_prep(*plan, 1)), "append 1");
  expect(dev.populated == 2, "populated is 2 when full");

  auto full = execute_decode_attention_prep(*plan, 2);
  expect(!full && full.error().code == qw38::runtime::ErrorCode::InvalidCapacity,
         "full cache is typed");
  expect(dev.populated == 2, "full-cache failure does not advance populated");

  auto still = execute_decode_attention_prep(*plan, 1);
  expect(!still &&
             still.error().code == qw38::runtime::ErrorCode::InvalidPopulatedLength,
         "stale position still mismatches when full");
}

void test_reset_session() {
  qw38::format::test::ScratchDir dir("qw38-attn-unit-reset");
  auto fx = qw38::runtime::test::write_language_fixture(dir.file("model.qw38"));
  expect(!fx.path.empty(), "write fixture");
  if (fx.path.empty()) {
    return;
  }
  auto rt = qw38::runtime::Runtime::create();
  expect(static_cast<bool>(rt), "runtime");
  if (!rt) {
    return;
  }
  auto model = rt->load(fx.path);
  expect(static_cast<bool>(model), "load fixture");
  if (!model) {
    return;
  }
  auto session = rt->create_session(*model, 4);
  expect(static_cast<bool>(session), "session");
  if (!session) {
    return;
  }
  expect(session->kv_populated(0) == 0, "new session populated is 0");
  expect(static_cast<bool>(session->set_populated_length(0, 3)), "set populated");
  expect(session->kv_populated(0) == 3, "populated set");
  std::uint16_t marker = 0xBEEF;
  auto off = kv_byte_offset(0, 0, 0, 1, 0, 4);
  expect(static_cast<bool>(off), "kv offset");
  if (off) {
    expect(static_cast<bool>(qw38::cuda::copy_h2d(
               static_cast<std::byte*>(session->kv().pointer) + *off, &marker,
               sizeof(marker), rt->stream())),
           "write marker");
  }
  expect(static_cast<bool>(session->reset()), "reset");
  expect(session->kv_populated(0) == 0, "reset clears populated");
  if (off) {
    std::uint16_t back = 0xFFFF;
    expect(static_cast<bool>(qw38::cuda::copy_d2h(
               &back, static_cast<std::byte*>(session->kv().pointer) + *off,
               sizeof(back), rt->stream())),
           "read marker");
    expect(static_cast<bool>(rt->stream().sync()), "sync reset");
    expect(back == 0, "reset zeros KV bytes");
  }
}

}  // namespace

int main() {
  auto stream = Stream::create();
  if (!stream) {
    fail("stream");
    return 1;
  }
  test_bind_errors(*stream);
  test_cuda_prepare_alias_errors(*stream);
  test_qg_ordering_and_suffix(*stream);
  test_capacity_and_mismatch(*stream);
  test_reset_session();
  if (g_failures != 0) {
    std::cerr << g_failures << " attention unit failures\n";
    return 1;
  }
  std::cout << "attention unit ok\n";
  return 0;
}
