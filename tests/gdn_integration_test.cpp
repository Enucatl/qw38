#include "gdn_support.hpp"

#include "compiler/quantization/reference.hpp"
#include "cuda/alloc.hpp"
#include "cuda/copy.hpp"
#include "format/format.hpp"
#include "runtime/runtime.hpp"
#include "runtime_support.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using qw38::cuda::malloc_count;
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
using qw38::gdn::test::expect;
using qw38::gdn::test::expect_bf16_close;
using qw38::gdn::test::expect_fp32_close;
using qw38::gdn::test::fail;
using qw38::gdn::test::fill_logical_pattern;
using qw38::gdn::test::g_failures;
using qw38::gdn::test::kConvHistoryTaps;
using qw38::gdn::test::kConvKernel;
using qw38::gdn::test::kDefaultRmsEps;
using qw38::gdn::test::kGdnConvBf16Abs;
using qw38::gdn::test::kGdnGateFp32Abs;
using qw38::gdn::test::kGdnGateFp32Rel;
using qw38::gdn::test::kGdnHeadDim;
using qw38::gdn::test::kGdnKeyHeads;
using qw38::gdn::test::kGdnQkFp32Abs;
using qw38::gdn::test::kGdnQkFp32Rel;
using qw38::gdn::test::kGdnRecurMultiOAbs;
using qw38::gdn::test::kGdnRecurMultiORel;
using qw38::gdn::test::kGdnRecurMultiSAbs;
using qw38::gdn::test::kGdnRecurMultiSRel;
using qw38::gdn::test::kGdnRecurOAbs;
using qw38::gdn::test::kGdnRecurORel;
using qw38::gdn::test::kGdnRecurSAbs;
using qw38::gdn::test::kGdnRecurSRel;
using qw38::gdn::test::kGdnSElemsPerLayer;
using qw38::gdn::test::kGdnValueHeads;
using qw38::gdn::test::kGdnZWidth;
using qw38::gdn::test::kHidden;
using qw38::gdn::test::kQkvWidth;
using qw38::gdn::test::make_logical;
using qw38::gdn::test::pack_bf16_tile;
using qw38::gdn::test::pack_q4_from_logical;
using qw38::gdn::test::pattern_h;
using qw38::gdn::test::residual_vec;
using qw38::gdn::test::v_from_convolved;
using qw38::gdn::test::zeros_f;
using qw38::reference::gdn_front_reference;
using qw38::reference::gdn_recurrence_step;
using qw38::runtime::Runtime;
using qw38::runtime::bind_gdn_front_plan;
using qw38::runtime::bind_gdn_recurrence_plan;
using qw38::runtime::execute_gdn_front;
using qw38::runtime::execute_gdn_recurrence;
using qw38::runtime::gdn_a_name;
using qw38::runtime::gdn_s_byte_offset;
using qw38::runtime::gdn_alog_name;
using qw38::runtime::gdn_b_name;
using qw38::runtime::gdn_conv_name;
using qw38::runtime::gdn_dt_name;
using qw38::runtime::gdn_norm_name;
using qw38::runtime::gdn_qkv_name;
using qw38::runtime::gdn_z_name;
using qw38::runtime::language_persistent_schema;

namespace {

struct HostFront {
  qw38::format::PackedMatrix qkv;
  qw38::format::PackedMatrix z;
  qw38::format::PackedMatrix a;
  qw38::format::PackedMatrix b;
  std::vector<std::uint16_t> w_qkv;
  std::vector<std::uint16_t> w_z;
  std::vector<std::uint16_t> w_a;
  std::vector<std::uint16_t> w_b;
  std::vector<std::uint16_t> gamma;
  std::vector<std::uint16_t> taps;
  std::vector<std::uint16_t> a_log;
  std::vector<std::uint16_t> dt_bias;
};

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

LogicalPhysicalMapping bf16_tile_mapping() {
  return LogicalPhysicalMapping{
      .kind = MappingKind::DenseTileNK,
      .tile_rows = 8,
      .tile_k = 256,
      .group_size = 0,
      .packed_bytes_per_tile_row = 512,
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

TensorRecord bf16_tile(std::uint32_t id, std::string name, std::uint64_t n,
                       std::uint64_t k) {
  TensorRecord t{};
  t.tensor_id = id;
  t.logical_name = std::move(name);
  t.shape = rank2(n, k);
  t.storage = StorageClass::Bf16;
  t.quantizer = LogicalQuantizerId::None;
  t.layout = PhysicalLayoutId::CudaBf16DenseTileV0;
  t.mapping = bf16_tile_mapping();
  return t;
}

TensorRecord bf16_vec(std::uint32_t id, std::string name, std::uint64_t n,
                      PhysicalLayoutId layout) {
  TensorRecord t{};
  t.tensor_id = id;
  t.logical_name = std::move(name);
  t.shape = rank1(n);
  t.storage = StorageClass::Bf16;
  t.quantizer = LogicalQuantizerId::None;
  t.layout = layout;
  t.mapping = LogicalPhysicalMapping{.kind = MappingKind::Identity};
  return t;
}

bool write_gdn_artifact(std::filesystem::path const& path, HostFront const& host) {
  ArtifactSchema schema{};
  schema.compiler = {.ident = "qw38-v0", .major = 0, .minor = 1, .patch = 0};
  schema.source_hash.bytes[0] = 0x11;
  schema.config_hash.bytes[0] = 0x22;
  schema.tokenizer_hash.bytes[0] = 0x33;
  schema.precision = v0_precision_policy();
  schema.scope = SemanticScope::PrimaryLanguage;

  auto gamma = bf16_vec(1, gdn_norm_name(0), kHidden,
                        PhysicalLayoutId::CudaBf16VectorV0);
  auto qkv = q4_weight(2, gdn_qkv_name(0), kQkvWidth, kHidden);
  auto z = q4_weight(3, gdn_z_name(0), kGdnZWidth, kHidden);
  auto a = bf16_tile(4, gdn_a_name(0), kGdnValueHeads, kHidden);
  auto b = bf16_tile(5, gdn_b_name(0), kGdnValueHeads, kHidden);
  TensorRecord taps{};
  taps.tensor_id = 6;
  taps.logical_name = gdn_conv_name(0);
  taps.shape = rank2(kConvKernel, kQkvWidth);
  taps.storage = StorageClass::Bf16;
  taps.quantizer = LogicalQuantizerId::None;
  taps.layout = PhysicalLayoutId::CudaBf16TapMajorV0;
  taps.mapping = LogicalPhysicalMapping{.kind = MappingKind::Identity};
  auto alog = bf16_vec(7, gdn_alog_name(0), kGdnValueHeads,
                       PhysicalLayoutId::CudaBf16VectorV0);
  auto dt = bf16_vec(8, gdn_dt_name(0), kGdnValueHeads,
                     PhysicalLayoutId::CudaBf16VectorV0);
  schema.tensors = {gamma, qkv, z, a, b, taps, alog, dt};
  schema.graph_bindings = {
      GraphBinding{.instance_id = 1,
                   .kind = SemanticNodeKind::GatedDeltaNet,
                   .role = TensorRole::NormGamma,
                   .layer_index = 0,
                   .tensor_id = 1},
      GraphBinding{.instance_id = 1,
                   .kind = SemanticNodeKind::GatedDeltaNet,
                   .role = TensorRole::DenseWeight,
                   .layer_index = 0,
                   .tensor_id = 2},
      GraphBinding{.instance_id = 1,
                   .kind = SemanticNodeKind::GatedDeltaNet,
                   .role = TensorRole::ConvWeight,
                   .layer_index = 0,
                   .tensor_id = 6},
      GraphBinding{.instance_id = 1,
                   .kind = SemanticNodeKind::GatedDeltaNet,
                   .role = TensorRole::TimeParameter,
                   .layer_index = 0,
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
  auto taps_b = qw38::activation::test::as_bytes(
      std::span<std::uint16_t const>(host.taps));
  auto alog_b = qw38::activation::test::as_bytes(
      std::span<std::uint16_t const>(host.a_log));
  auto dt_b = qw38::activation::test::as_bytes(
      std::span<std::uint16_t const>(host.dt_bias));
  if (!write(gamma.logical_name, SpanKind::Payload, gamma_b) ||
      !write(qkv.logical_name, SpanKind::Payload, host.qkv.codes) ||
      !write(qkv.logical_name, SpanKind::Scales, host.qkv.scales) ||
      !write(z.logical_name, SpanKind::Payload, host.z.codes) ||
      !write(z.logical_name, SpanKind::Scales, host.z.scales) ||
      !write(a.logical_name, SpanKind::Payload, host.a.codes) ||
      !write(b.logical_name, SpanKind::Payload, host.b.codes) ||
      !write(taps.logical_name, SpanKind::Payload, taps_b) ||
      !write(alog.logical_name, SpanKind::Payload, alog_b) ||
      !write(dt.logical_name, SpanKind::Payload, dt_b)) {
    return false;
  }
  auto id = writer->finalize();
  if (!id) {
    fail(std::string("finalize: ") + qw38::format::error_message(id.error()));
    return false;
  }
  return true;
}

bool make_host(HostFront& host) {
  auto lq = make_logical(LogicalQuantizerId::Q4G64V0, kQkvWidth, kHidden, 2, 0x3C00);
  auto lz = make_logical(LogicalQuantizerId::Q4G64V0, kGdnZWidth, kHidden, -1, 0x3C00);
  fill_logical_pattern(lq, 4);
  fill_logical_pattern(lz, 8);
  if (!pack_q4_from_logical(lq, host.qkv, "qkv") ||
      !pack_q4_from_logical(lz, host.z, "z")) {
    return false;
  }
  auto wq = qw38::compiler::dequantize_to_bf16(lq);
  auto wz = qw38::compiler::dequantize_to_bf16(lz);
  if (!wq || !wz) {
    fail("dequant");
    return false;
  }
  host.w_qkv = std::move(*wq);
  host.w_z = std::move(*wz);
  host.w_a = pattern_h(kGdnValueHeads * kHidden, 0.05f);
  host.w_b = pattern_h(kGdnValueHeads * kHidden, -0.04f);
  if (!pack_bf16_tile(host.w_a, kGdnValueHeads, kHidden, host.a, "a") ||
      !pack_bf16_tile(host.w_b, kGdnValueHeads, kHidden, host.b, "b")) {
    return false;
  }
  host.gamma = pattern_h(kHidden, 0.11f);
  host.taps = pattern_h(kConvKernel * kQkvWidth, 0.22f);
  host.a_log = pattern_h(kGdnValueHeads, -0.3f);
  host.dt_bias = pattern_h(kGdnValueHeads, 0.45f);
  return true;
}

}  // namespace

int main() {
  HostFront host;
  if (!make_host(host)) {
    return 1;
  }

  qw38::format::test::ScratchDir dir("qw38-gdn-int");
  auto path = dir.file("gdn.qw38");
  if (!write_gdn_artifact(path, host)) {
    return 1;
  }

  auto rt = Runtime::create();
  expect(static_cast<bool>(rt), "Runtime::create");
  if (!rt) {
    return 1;
  }
  auto model = rt->load(path);
  if (!model) {
    fail("load: " + qw38::runtime::error_message(model.error()));
    return 1;
  }
  auto s_cont = rt->create_session(*model, 1);
  auto s_snap = rt->create_session(*model, 1);
  if (!s_cont || !s_snap) {
    fail("session");
    return 1;
  }

  auto plan_cont = bind_gdn_front_plan(*model, *s_cont, 0, rt->stream());
  auto plan_snap = bind_gdn_front_plan(*model, *s_snap, 0, rt->stream());
  auto recur_cont = bind_gdn_recurrence_plan(*s_cont, 0, rt->stream());
  auto recur_snap = bind_gdn_recurrence_plan(*s_snap, 0, rt->stream());
  if (!plan_cont || !plan_snap || !recur_cont || !recur_snap) {
    fail("bind");
    return 1;
  }
  expect(plan_cont->scratch.q_hat.extent[0] == kGdnKeyHeads, "16 q heads");
  expect(plan_cont->history.extent[0] == kConvHistoryTaps, "history [3,10240]");
  expect(plan_cont->scratch.o.extent[0] == kGdnValueHeads, "o [48,128]");
  expect(recur_cont->s.dtype == qw38::format::ArithmeticDtype::Fp32, "S is FP32");

  std::vector<std::uint16_t> cpu_hist(kConvHistoryTaps * kQkvWidth,
                                      qw38::format::fp32_to_bf16_rne(0.0f));
  std::uint32_t cpu_cursor = 0;
  qw38::reference::GdnFrontReference last{};
  auto cpu_s = zeros_f(static_cast<std::uint32_t>(kGdnSElemsPerLayer));
  auto cpu_o = zeros_f(kGdnValueHeads * kGdnHeadDim);

  for (int step = 0; step < 5; ++step) {
    auto residual = residual_vec(kHidden, 0.3f + 0.1f * static_cast<float>(step));
    auto bytes = qw38::activation::test::as_bytes(std::span<float const>(residual));
    auto up1 = qw38::cuda::copy_h2d(s_cont->residual_h().pointer, bytes.data(),
                                    bytes.size(), rt->stream());
    auto up2 = qw38::cuda::copy_h2d(s_snap->residual_h().pointer, bytes.data(),
                                    bytes.size(), rt->stream());
    expect(static_cast<bool>(up1) && static_cast<bool>(up2), "upload residual");
    auto cpu = gdn_front_reference(residual, host.gamma, kDefaultRmsEps, host.w_qkv,
                                   host.w_z, host.w_a, host.w_b, host.taps,
                                   host.a_log, host.dt_bias, cpu_hist, cpu_cursor);
    if (!cpu) {
      fail("cpu step");
      return 1;
    }
    cpu_hist = cpu->history;
    cpu_cursor = cpu->cursor;
    auto v_cpu = v_from_convolved(cpu->convolved);
    auto rst_cpu = gdn_recurrence_step(cpu->q_hat, cpu->k_hat, cpu->alpha, cpu->beta,
                                       v_cpu, cpu_s, cpu_o);
    if (!rst_cpu) {
      fail("cpu recurrence");
      return 1;
    }
    last = std::move(*cpu);

    auto const mallocs = malloc_count();
    auto st = execute_gdn_front(*plan_cont);
    if (!st) {
      fail("execute cont: " + qw38::runtime::error_message(st.error()));
      return 1;
    }
    auto rst = execute_gdn_recurrence(*recur_cont);
    if (!rst) {
      fail("recur cont: " + qw38::runtime::error_message(rst.error()));
      return 1;
    }
    expect(malloc_count() == mallocs, "execute allocates nothing");
    if (step < 3) {
      auto st2 = execute_gdn_front(*plan_snap);
      if (!st2) {
        fail("execute snap: " + qw38::runtime::error_message(st2.error()));
        return 1;
      }
      auto rst2 = execute_gdn_recurrence(*recur_snap);
      if (!rst2) {
        fail("recur snap: " + qw38::runtime::error_message(rst2.error()));
        return 1;
      }
    }
    if (step == 2) {
      auto snap = s_snap->save();
      if (!snap) {
        fail("save");
        return 1;
      }
      auto restored = rt->create_session(*model, 1);
      if (!restored) {
        fail("restore session");
        return 1;
      }
      auto rst_s = restored->restore(*snap);
      if (!rst_s) {
        fail("restore: " + qw38::runtime::error_message(rst_s.error()));
        return 1;
      }
      s_snap = std::move(*restored);
      auto rebound = bind_gdn_front_plan(*model, *s_snap, 0, rt->stream());
      auto rebound_r = bind_gdn_recurrence_plan(*s_snap, 0, rt->stream());
      if (!rebound || !rebound_r) {
        fail("rebind");
        return 1;
      }
      plan_snap = std::move(*rebound);
      recur_snap = std::move(*rebound_r);
      expect(s_snap->conv_cursor()[0] == cpu_cursor, "restored cursor");
    }
    if (step >= 3) {
      auto st2 = execute_gdn_front(*plan_snap);
      if (!st2) {
        fail("execute restored: " + qw38::runtime::error_message(st2.error()));
        return 1;
      }
      auto rst2 = execute_gdn_recurrence(*recur_snap);
      if (!rst2) {
        fail("recur restored: " + qw38::runtime::error_message(rst2.error()));
        return 1;
      }
    }
  }
  expect(s_cont->conv_cursor()[0] == cpu_cursor, "uninterrupted cursor");
  expect(s_snap->conv_cursor()[0] == cpu_cursor, "restored continuation cursor");

  std::size_t const nqk = static_cast<std::size_t>(kGdnKeyHeads) * kGdnHeadDim;
  std::vector<float> q_cont(nqk);
  std::vector<float> q_snap(nqk);
  std::vector<std::uint16_t> c_cont(kQkvWidth);
  std::vector<std::uint16_t> c_snap(kQkvWidth);
  std::vector<float> a_cont(kGdnValueHeads);
  std::vector<float> a_snap(kGdnValueHeads);
  auto dl = [&](void* dst, void const* src, std::uint64_t n) {
    return static_cast<bool>(
        qw38::cuda::copy_d2h(dst, src, n, rt->stream()));
  };
  if (!dl(q_cont.data(), plan_cont->scratch.q_hat.pointer, nqk * 4) ||
      !dl(q_snap.data(), plan_snap->scratch.q_hat.pointer, nqk * 4) ||
      !dl(c_cont.data(), plan_cont->scratch.convolved.pointer, kQkvWidth * 2) ||
      !dl(c_snap.data(), plan_snap->scratch.convolved.pointer, kQkvWidth * 2) ||
      !dl(a_cont.data(), plan_cont->scratch.alpha.pointer, kGdnValueHeads * 4) ||
      !dl(a_snap.data(), plan_snap->scratch.alpha.pointer, kGdnValueHeads * 4) ||
      !rt->stream().sync()) {
    fail("download stages");
    return 1;
  }
  expect_fp32_close(q_cont, last.q_hat, "cont q_hat", kGdnQkFp32Abs, kGdnQkFp32Rel);
  expect_fp32_close(q_snap, last.q_hat, "snap q_hat", kGdnQkFp32Abs, kGdnQkFp32Rel);
  expect_bf16_close(c_cont, last.convolved, "cont conv", kGdnConvBf16Abs, 1.0e-4f);
  expect_bf16_close(c_snap, last.convolved, "snap conv", kGdnConvBf16Abs, 1.0e-4f);
  expect_fp32_close(a_cont, last.alpha, "cont alpha", kGdnGateFp32Abs, kGdnGateFp32Rel);
  expect_fp32_close(a_snap, last.alpha, "snap alpha", kGdnGateFp32Abs, kGdnGateFp32Rel);

  auto hist_off = qw38::runtime::conv_history_byte_offset(0, 0, 0);
  expect(static_cast<bool>(hist_off), "hist offset");
  std::vector<std::uint16_t> h_cont(kConvHistoryTaps * kQkvWidth);
  std::vector<std::uint16_t> h_snap(kConvHistoryTaps * kQkvWidth);
  auto* hp_cont = static_cast<std::byte*>(s_cont->conv_history().pointer) + *hist_off;
  auto* hp_snap = static_cast<std::byte*>(s_snap->conv_history().pointer) + *hist_off;
  if (!dl(h_cont.data(), hp_cont, h_cont.size() * 2) ||
      !dl(h_snap.data(), hp_snap, h_snap.size() * 2) || !rt->stream().sync()) {
    fail("history download");
    return 1;
  }
  expect_bf16_close(h_cont, last.history, "cont history", 0.0f, 0.0f);
  expect_bf16_close(h_snap, last.history, "snap history", 0.0f, 0.0f);

  auto s_off = gdn_s_byte_offset(0, 0, 0, 0);
  expect(static_cast<bool>(s_off), "S offset");
  std::size_t const nv = static_cast<std::size_t>(kGdnValueHeads) * kGdnHeadDim;
  std::vector<float> state_cont(kGdnSElemsPerLayer);
  std::vector<float> state_snap(kGdnSElemsPerLayer);
  std::vector<float> o_cont(nv);
  std::vector<float> o_snap(nv);
  if (!s_off ||
      !dl(state_cont.data(),
          static_cast<std::byte*>(s_cont->gdn_s().pointer) + *s_off,
          state_cont.size() * 4) ||
      !dl(state_snap.data(),
          static_cast<std::byte*>(s_snap->gdn_s().pointer) + *s_off,
          state_snap.size() * 4) ||
      !dl(o_cont.data(), plan_cont->scratch.o.pointer, nv * 4) ||
      !dl(o_snap.data(), plan_snap->scratch.o.pointer, nv * 4) ||
      !rt->stream().sync()) {
    fail("S/o download");
    return 1;
  }
  expect_fp32_close(state_cont, state_snap, "cont vs snap S", kGdnRecurSAbs,
                    kGdnRecurSRel);
  expect_fp32_close(o_cont, o_snap, "cont vs snap o", kGdnRecurOAbs, kGdnRecurORel);
  expect_fp32_close(state_cont, cpu_s, "cont vs cpu S", kGdnRecurMultiSAbs,
                    kGdnRecurMultiSRel);
  expect_fp32_close(state_snap, cpu_s, "snap vs cpu S", kGdnRecurMultiSAbs,
                    kGdnRecurMultiSRel);
  expect_fp32_close(o_cont, cpu_o, "cont vs cpu o", kGdnRecurMultiOAbs,
                    kGdnRecurMultiORel);
  expect_fp32_close(o_snap, cpu_o, "snap vs cpu o", kGdnRecurMultiOAbs,
                    kGdnRecurMultiORel);

  if (g_failures != 0) {
    std::cerr << g_failures << " gdn integration failures\n";
    return 1;
  }
  std::cout << "gdn integration ok\n";
  return 0;
}
