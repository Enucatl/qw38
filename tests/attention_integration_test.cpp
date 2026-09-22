#include "attention_support.hpp"

#include "cuda/alloc.hpp"
#include "cuda/copy.hpp"
#include "format/format.hpp"
#include "runtime/runtime.hpp"
#include "runtime_support.hpp"

#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <vector>

using qw38::attn::test::HostAttn;
using qw38::attn::test::download_vec;
using qw38::attn::test::expect;
using qw38::attn::test::expect_bf16_close;
using qw38::attn::test::fail;
using qw38::attn::test::g_failures;
using qw38::attn::test::kAttnKvWidth;
using qw38::attn::test::kAttnPrepSmallAbs;
using qw38::attn::test::kAttnPrepSmallRel;
using qw38::attn::test::kHeadDim;
using qw38::attn::test::kHidden;
using qw38::attn::test::kKvHeads;
using qw38::attn::test::kQgWidth;
using qw38::attn::test::kQueryHeads;
using qw38::attn::test::make_host_q4;
using qw38::format::ArithmeticDtype;
using qw38::format::ArtifactSchema;
using qw38::format::ArtifactWriter;
using qw38::format::GraphBinding;
using qw38::format::LogicalPhysicalMapping;
using qw38::format::LogicalQuantizerId;
using qw38::format::MappingKind;
using qw38::format::PhysicalLayoutId;
using qw38::format::SemanticNodeKind;
using qw38::format::SemanticScope;
using qw38::format::SpanKind;
using qw38::format::StorageClass;
using qw38::format::TensorRecord;
using qw38::format::TensorRole;
using qw38::format::v0_precision_policy;
using qw38::reference::attn_prep_reference;
using qw38::runtime::Runtime;
using qw38::runtime::attn_k_name;
using qw38::runtime::attn_k_norm_name;
using qw38::runtime::attn_norm_name;
using qw38::runtime::attn_q_name;
using qw38::runtime::attn_q_norm_name;
using qw38::runtime::attn_v_name;
using qw38::runtime::bind_attention_prep_plan;
using qw38::runtime::execute_decode_attention_prep;
using qw38::runtime::kv_byte_offset;
using qw38::runtime::language_persistent_schema;

namespace {

qw38::format::TensorShape rank1(std::uint64_t n) {
  qw38::format::TensorShape s{};
  s.rank = 1;
  s.logical[0] = n;
  s.padded[0] = n;
  return s;
}

qw38::format::TensorShape rank2(std::uint64_t n, std::uint64_t k) {
  qw38::format::TensorShape s{};
  s.rank = 2;
  s.logical[0] = n;
  s.logical[1] = k;
  s.padded[0] = n;
  s.padded[1] = k;
  return s;
}

LogicalPhysicalMapping q4_mapping() {
  return LogicalPhysicalMapping{
      .kind = MappingKind::DenseTileNK,
      .tile_rows = 8,
      .tile_k = 256,
      .group_size = 64,
      .packed_bytes_per_tile_row = 128,
  };
}

TensorRecord q4_weight(std::uint32_t id, std::string name, std::uint64_t n,
                       std::uint64_t k) {
  TensorRecord t{};
  t.tensor_id = id;
  t.logical_name = std::move(name);
  t.shape = rank2(n, k);
  t.storage = StorageClass::Int4Grouped;
  t.quantizer = LogicalQuantizerId::Q4G64V0;
  t.layout = PhysicalLayoutId::CudaQ4G64V0;
  t.mapping = q4_mapping();
  return t;
}

TensorRecord bf16_vec(std::uint32_t id, std::string name, std::uint64_t n) {
  TensorRecord t{};
  t.tensor_id = id;
  t.logical_name = std::move(name);
  t.shape = rank1(n);
  t.storage = StorageClass::Bf16;
  t.quantizer = LogicalQuantizerId::None;
  t.layout = PhysicalLayoutId::CudaBf16VectorV0;
  t.mapping = LogicalPhysicalMapping{.kind = MappingKind::Identity};
  return t;
}

bool write_attn_artifact(std::filesystem::path const& path, HostAttn const& host) {
  ArtifactSchema schema{};
  schema.compiler = {.ident = "qw38-v0", .major = 0, .minor = 1, .patch = 0};
  schema.source_hash.bytes[0] = 0x41;
  schema.config_hash.bytes[0] = 0x42;
  schema.tokenizer_hash.bytes[0] = 0x43;
  schema.precision = v0_precision_policy();
  schema.scope = SemanticScope::PrimaryLanguage;

  constexpr std::uint32_t kLayer = 3;
  auto gamma = bf16_vec(1, attn_norm_name(kLayer), kHidden);
  auto qg = q4_weight(2, attn_q_name(kLayer), kQgWidth, kHidden);
  auto k = q4_weight(3, attn_k_name(kLayer), kAttnKvWidth, kHidden);
  auto v = q4_weight(4, attn_v_name(kLayer), kAttnKvWidth, kHidden);
  auto qn = bf16_vec(5, attn_q_norm_name(kLayer), kHeadDim);
  auto kn = bf16_vec(6, attn_k_norm_name(kLayer), kHeadDim);
  TensorRecord rope{};
  rope.tensor_id = 7;
  rope.logical_name = "rope.inv_freq";
  rope.shape = rank1(32);
  rope.storage = StorageClass::Fp32;
  rope.quantizer = LogicalQuantizerId::None;
  rope.layout = PhysicalLayoutId::CudaFp32VectorV0;
  rope.mapping = LogicalPhysicalMapping{.kind = MappingKind::Identity};
  schema.tensors = {gamma, qg, k, v, qn, kn, rope};
  schema.graph_bindings = {
      GraphBinding{.instance_id = 1,
                   .kind = SemanticNodeKind::GatedAttention,
                   .role = TensorRole::NormGamma,
                   .layer_index = kLayer,
                   .tensor_id = 1},
      GraphBinding{.instance_id = 1,
                   .kind = SemanticNodeKind::GatedAttention,
                   .role = TensorRole::DenseWeight,
                   .layer_index = kLayer,
                   .tensor_id = 2},
      GraphBinding{.instance_id = 1,
                   .kind = SemanticNodeKind::GatedAttention,
                   .role = TensorRole::VectorWeight,
                   .layer_index = kLayer,
                   .tensor_id = 7},
  };
  auto state = language_persistent_schema();
  schema.state.assign(state.begin(), state.end());
  schema.scratch = qw38::runtime::test::language_scratch();

  auto writer = ArtifactWriter::create(path, schema);
  if (!writer) {
    fail(std::string("writer create: ") +
         qw38::format::error_message(writer.error()));
    return false;
  }
  auto write = [&](std::string_view name, SpanKind kind,
                   std::vector<std::byte> const& bytes) {
    auto st = writer->write_span(name, kind, bytes);
    if (!st) {
      fail(std::string("write ") + std::string(name) + ": " +
           qw38::format::error_message(st.error()));
      return false;
    }
    return true;
  };
  auto gamma_b = qw38::activation::test::as_bytes(
      std::span<std::uint16_t const>(host.gamma));
  auto qn_b = qw38::activation::test::as_bytes(
      std::span<std::uint16_t const>(host.gamma_q));
  auto kn_b = qw38::activation::test::as_bytes(
      std::span<std::uint16_t const>(host.gamma_k));
  auto inv_b = qw38::activation::test::as_bytes(
      std::span<float const>(host.inv_freq.data(), host.inv_freq.size()));
  if (!write(gamma.logical_name, SpanKind::Payload, gamma_b) ||
      !write(qg.logical_name, SpanKind::Payload, host.qg.codes) ||
      !write(qg.logical_name, SpanKind::Scales, host.qg.scales) ||
      !write(k.logical_name, SpanKind::Payload, host.k.codes) ||
      !write(k.logical_name, SpanKind::Scales, host.k.scales) ||
      !write(v.logical_name, SpanKind::Payload, host.v.codes) ||
      !write(v.logical_name, SpanKind::Scales, host.v.scales) ||
      !write(qn.logical_name, SpanKind::Payload, qn_b) ||
      !write(kn.logical_name, SpanKind::Payload, kn_b) ||
      !write(rope.logical_name, SpanKind::Payload, inv_b)) {
    return false;
  }
  auto id = writer->finalize();
  if (!id) {
    fail(std::string("finalize: ") + qw38::format::error_message(id.error()));
    return false;
  }
  return true;
}

bool upload_residual(qw38::runtime::Session& session, std::span<float const> h,
                     qw38::cuda::Stream const& stream, std::string_view tag) {
  auto st = qw38::cuda::copy_h2d(session.residual_h().pointer, h.data(),
                                 h.size() * sizeof(float), stream);
  if (!st) {
    fail(std::string(tag) + " residual upload");
    return false;
  }
  return true;
}

}  // namespace

int main() {
  HostAttn host;
  if (!make_host_q4(host, 7)) {
    return 1;
  }
  qw38::format::test::ScratchDir dir("qw38-attn-integration");
  auto path = dir.file("attn.qw38");
  if (!write_attn_artifact(path, host)) {
    return 1;
  }

  auto rt = Runtime::create();
  if (!rt) {
    fail("runtime");
    return 1;
  }
  auto model = rt->load(path);
  if (!model) {
    fail("load: " + qw38::runtime::error_message(model.error()));
    return 1;
  }
  constexpr std::uint64_t kCap = 8;
  auto s1 = rt->create_session(*model, kCap);
  auto s2 = rt->create_session(*model, kCap);
  if (!s1 || !s2) {
    fail("sessions");
    return 1;
  }
  expect(s1->kv().pointer != s2->kv().pointer, "sessions have isolated KV");
  expect(s1->kv_populated() == 0 && s2->kv_populated() == 0, "fresh populated");

  auto p1 = bind_attention_prep_plan(*model, *s1, 3, rt->stream());
  auto p2 = bind_attention_prep_plan(*model, *s2, 3, rt->stream());
  if (!p1 || !p2) {
    fail("bind plans");
    return 1;
  }

  auto r2 = host.residual;
  for (auto& x : r2) {
    x = -x;
  }
  if (!upload_residual(*s1, host.residual, rt->stream(), "s1") ||
      !upload_residual(*s2, r2, rt->stream(), "s2")) {
    return 1;
  }

  auto malloc_before = qw38::cuda::malloc_count();
  std::vector<qw38::reference::AttnPrepReference> cpu1;
  for (std::uint64_t t = 0; t < 3; ++t) {
    auto cpu = attn_prep_reference(host.residual, host.gamma,
                                   qw38::reference::kDefaultRmsEps, host.w_qg,
                                   host.w_k, host.w_v, host.gamma_q, host.gamma_k,
                                   host.inv_freq, static_cast<std::int32_t>(t));
    if (!cpu) {
      fail("cpu step");
      return 1;
    }
    cpu1.push_back(std::move(*cpu));
    auto st = execute_decode_attention_prep(*p1, t);
    if (!st) {
      fail("s1 append " + std::to_string(t) + ": " +
           qw38::runtime::error_message(st.error()));
      return 1;
    }
    expect(s1->kv_populated() == t + 1, "s1 populated advances after success");
  }
  expect(s2->kv_populated() == 0, "s2 populated is unaffected by s1");

  auto snap = s1->save();
  if (!snap) {
    fail("save");
    return 1;
  }
  expect(snap->kv_populated == 3, "snapshot populated");

  auto st2 = execute_decode_attention_prep(*p2, 0);
  if (!st2) {
    fail("s2 append: " + qw38::runtime::error_message(st2.error()));
    return 1;
  }
  expect(s2->kv_populated() == 1, "s2 append independent");
  expect(qw38::cuda::malloc_count() == malloc_before,
         "hot-path execute does not allocate");

  auto restored = rt->create_session(*model, kCap);
  if (!restored) {
    fail("restore session");
    return 1;
  }
  auto rst = restored->restore(*snap);
  if (!rst) {
    fail("restore: " + qw38::runtime::error_message(rst.error()));
    return 1;
  }
  expect(restored->kv_populated() == 3, "restored populated");
  auto pr = bind_attention_prep_plan(*model, *restored, 3, rt->stream());
  if (!pr) {
    fail("rebind restored");
    return 1;
  }
  if (!upload_residual(*restored, host.residual, rt->stream(), "restored")) {
    return 1;
  }
  auto st3 = execute_decode_attention_prep(*pr, 3);
  if (!st3) {
    fail("restored append 3: " + qw38::runtime::error_message(st3.error()));
    return 1;
  }
  expect(restored->kv_populated() == 4, "restored continuation");
  expect(s1->kv_populated() == 3, "source session populated unchanged by restore");

  std::vector<std::uint16_t> host_kv(
      static_cast<std::size_t>(16) * 2u * kKvHeads * kCap * kHeadDim);
  expect(static_cast<bool>(qw38::cuda::copy_d2h(
             host_kv.data(), s1->kv().pointer, host_kv.size() * 2, rt->stream())),
         "download s1 kv");
  std::vector<std::uint16_t> host_kv2(host_kv.size());
  expect(static_cast<bool>(qw38::cuda::copy_d2h(
             host_kv2.data(), s2->kv().pointer, host_kv2.size() * 2, rt->stream())),
         "download s2 kv");
  std::vector<std::uint16_t> host_kvr(host_kv.size());
  expect(static_cast<bool>(qw38::cuda::copy_d2h(
             host_kvr.data(), restored->kv().pointer, host_kvr.size() * 2,
             rt->stream())),
         "download restored kv");
  expect(static_cast<bool>(rt->stream().sync()), "sync downloads");

  for (std::uint64_t t = 0; t < 3; ++t) {
    for (std::uint32_t h = 0; h < kKvHeads; ++h) {
      auto k_off = kv_byte_offset(0, 0, h, t, 0, kCap);
      auto v_off = kv_byte_offset(0, 1, h, t, 0, kCap);
      expect(static_cast<bool>(k_off) && static_cast<bool>(v_off), "index");
      if (!k_off || !v_off) {
        return 1;
      }
      std::size_t const ke = *k_off / 2;
      std::size_t const ve = *v_off / 2;
      expect_bf16_close(
          std::span<std::uint16_t const>(host_kv.data() + ke, kHeadDim),
          std::span<std::uint16_t const>(cpu1[static_cast<std::size_t>(t)].k.data() +
                                             h * kHeadDim,
                                         kHeadDim),
          "s1 cached K token " + std::to_string(t), kAttnPrepSmallAbs,
          kAttnPrepSmallRel);
      expect_bf16_close(
          std::span<std::uint16_t const>(host_kv.data() + ve, kHeadDim),
          std::span<std::uint16_t const>(cpu1[static_cast<std::size_t>(t)].v.data() +
                                             h * kHeadDim,
                                         kHeadDim),
          "s1 cached V token " + std::to_string(t), kAttnPrepSmallAbs,
          kAttnPrepSmallRel);
      expect(host_kvr[ke] == host_kv[ke], "restored K matches snapshot bytes");
    }
  }

  bool isolated = false;
  for (std::size_t i = 0; i < host_kv.size(); ++i) {
    if (host_kv[i] != host_kv2[i]) {
      isolated = true;
      break;
    }
  }
  expect(isolated, "two sessions store distinct KV bytes");

  std::vector<std::uint16_t> q_s1(kQueryHeads * kHeadDim);
  std::vector<std::uint16_t> g_s1(kQueryHeads * kHeadDim);
  auto ws = s1->scratch(qw38::format::ScratchKind::AttentionWorkspace);
  expect(static_cast<bool>(ws), "s1 workspace");
  if (ws) {
    expect(static_cast<bool>(qw38::cuda::copy_d2h(
               q_s1.data(),
               static_cast<std::byte*>(ws->pointer) + qw38::runtime::kAttnOffQ,
               q_s1.size() * 2, rt->stream())),
           "s1 Q");
    expect(static_cast<bool>(qw38::cuda::copy_d2h(
               g_s1.data(),
               static_cast<std::byte*>(ws->pointer) + qw38::runtime::kAttnOffG,
               g_s1.size() * 2, rt->stream())),
           "s1 g");
    expect(static_cast<bool>(rt->stream().sync()), "sync Q/g");
    expect_bf16_close(q_s1, cpu1[2].q, "last prepared Q", kAttnPrepSmallAbs,
                      kAttnPrepSmallRel);
    expect_bf16_close(g_s1, cpu1[2].g, "last prepared g", kAttnPrepSmallAbs,
                      kAttnPrepSmallRel);
  }

  expect(s1->reset() && s1->kv_populated() == 0, "session reset");
  std::uint16_t marker = 0xFFFF;
  expect(static_cast<bool>(qw38::cuda::copy_d2h(
             &marker, s1->kv().pointer, sizeof(marker), rt->stream())),
         "read reset kv");
  expect(static_cast<bool>(rt->stream().sync()), "sync reset kv");
  expect(marker == 0, "reset zeros cache bytes");

  if (g_failures != 0) {
    std::cerr << g_failures << " attention integration failures\n";
    return 1;
  }
  std::cout << "attention integration ok\n";
  return 0;
}
