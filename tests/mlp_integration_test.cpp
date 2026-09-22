#include "mlp_support.hpp"

#include "format/format.hpp"
#include "runtime/runtime.hpp"
#include "runtime_support.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using qw38::cuda::Stream;
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
using qw38::mlp::test::HostMlp;
using qw38::mlp::test::bf16_vec;
using qw38::mlp::test::download_vec;
using qw38::mlp::test::expect;
using qw38::mlp::test::expect_fp32_close;
using qw38::mlp::test::fail;
using qw38::mlp::test::fill_logical_pattern;
using qw38::mlp::test::g_failures;
using qw38::mlp::test::kResAbs;
using qw38::mlp::test::kResRel;
using qw38::mlp::test::make_logical;
using qw38::mlp::test::pack_q4_from_logical;
using qw38::mlp::test::residual_vec;
using qw38::reference::decode_mlp_reference;
using qw38::reference::kDefaultRmsEps;
using qw38::reference::kFfn;
using qw38::reference::kHidden;
using qw38::runtime::Runtime;
using qw38::runtime::bind_mlp_plan;
using qw38::runtime::execute_decode_mlp;
using qw38::runtime::language_persistent_schema;
using qw38::runtime::mlp_down_name;
using qw38::runtime::mlp_gate_name;
using qw38::runtime::mlp_norm_name;
using qw38::runtime::mlp_up_name;
using qw38::runtime::test::language_scratch;

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

bool write_mlp_artifact(std::filesystem::path const& path, HostMlp const& host) {
  ArtifactSchema schema{};
  schema.compiler = {.ident = "qw38-v0", .major = 0, .minor = 1, .patch = 0};
  schema.source_hash.bytes[0] = 0x10;
  schema.config_hash.bytes[0] = 0x20;
  schema.tokenizer_hash.bytes[0] = 0x30;
  schema.precision = v0_precision_policy();
  schema.scope = SemanticScope::PrimaryLanguage;

  TensorRecord gamma{};
  gamma.tensor_id = 1;
  gamma.logical_name = mlp_norm_name(0);
  gamma.shape = rank1(kHidden);
  gamma.storage = StorageClass::Bf16;
  gamma.quantizer = LogicalQuantizerId::None;
  gamma.layout = PhysicalLayoutId::CudaBf16VectorV0;
  gamma.mapping = LogicalPhysicalMapping{.kind = MappingKind::Identity};

  auto gate = q4_weight(2, mlp_gate_name(0), kFfn, kHidden);
  auto up = q4_weight(3, mlp_up_name(0), kFfn, kHidden);
  auto down = q4_weight(4, mlp_down_name(0), kHidden, kFfn);
  schema.tensors = {gamma, gate, up, down};
  schema.graph_bindings = {
      GraphBinding{.instance_id = 2,
                   .kind = SemanticNodeKind::Mlp,
                   .role = TensorRole::NormGamma,
                   .layer_index = 0,
                   .tensor_id = 1},
      GraphBinding{.instance_id = 2,
                   .kind = SemanticNodeKind::Mlp,
                   .role = TensorRole::DenseWeight,
                   .layer_index = 0,
                   .tensor_id = 2},
      GraphBinding{.instance_id = 2,
                   .kind = SemanticNodeKind::Mlp,
                   .role = TensorRole::DenseWeight,
                   .layer_index = 0,
                   .tensor_id = 3},
      GraphBinding{.instance_id = 2,
                   .kind = SemanticNodeKind::Mlp,
                   .role = TensorRole::DenseWeight,
                   .layer_index = 0,
                   .tensor_id = 4},
  };
  auto state = language_persistent_schema();
  schema.state.assign(state.begin(), state.end());
  schema.scratch = language_scratch();

  auto writer = ArtifactWriter::create(path, schema);
  if (!writer) {
    fail(std::string("writer create: ") +
         qw38::format::error_message(writer.error()));
    return false;
  }
  auto gamma_bytes = qw38::activation::test::as_bytes(std::span<std::uint16_t const>(
      host.gamma));
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
  if (!write(gamma.logical_name, SpanKind::Payload, gamma_bytes) ||
      !write(gate.logical_name, SpanKind::Payload, host.gate.codes) ||
      !write(gate.logical_name, SpanKind::Scales, host.gate.scales) ||
      !write(up.logical_name, SpanKind::Payload, host.up.codes) ||
      !write(up.logical_name, SpanKind::Scales, host.up.scales) ||
      !write(down.logical_name, SpanKind::Payload, host.down.codes) ||
      !write(down.logical_name, SpanKind::Scales, host.down.scales)) {
    return false;
  }
  auto id = writer->finalize();
  if (!id) {
    fail(std::string("finalize: ") + qw38::format::error_message(id.error()));
    return false;
  }
  return true;
}

}  // namespace

int main() {
  HostMlp host;
  auto lg = make_logical(LogicalQuantizerId::Q4G64V0, kFfn, kHidden, 1, 0x3C00);
  auto lu = make_logical(LogicalQuantizerId::Q4G64V0, kFfn, kHidden, -3, 0x3C00);
  auto ld = make_logical(LogicalQuantizerId::Q4G64V0, kHidden, kFfn, 2, 0x3C00);
  fill_logical_pattern(lg, 3);
  fill_logical_pattern(lu, 6);
  fill_logical_pattern(ld, 11);
  if (!pack_q4_from_logical(lg, host.gate, "int gate") ||
      !pack_q4_from_logical(lu, host.up, "int up") ||
      !pack_q4_from_logical(ld, host.down, "int down")) {
    return 1;
  }
  auto wg = qw38::compiler::dequantize_to_bf16(lg);
  auto wu = qw38::compiler::dequantize_to_bf16(lu);
  auto wd = qw38::compiler::dequantize_to_bf16(ld);
  if (!wg || !wu || !wd) {
    fail("dequant");
    return 1;
  }
  host.w_gate = std::move(*wg);
  host.w_up = std::move(*wu);
  host.w_down = std::move(*wd);
  host.gamma = bf16_vec(kHidden, 0.09f);
  host.h_mid = residual_vec(kHidden, 0.41f);

  auto cpu1 = decode_mlp_reference(host.h_mid, host.gamma, kDefaultRmsEps,
                                   host.w_gate, host.w_up, host.w_down);
  if (!cpu1) {
    fail("cpu1: " + qw38::reference::error_message(cpu1.error()));
    return 1;
  }
  qw38::format::test::ScratchDir dir("qw38-mlp-int");
  auto path = dir.file("mlp.qw38");
  if (!write_mlp_artifact(path, host)) {
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
  auto session = rt->create_session(*model, 1);
  if (!session) {
    fail("session: " + qw38::runtime::error_message(session.error()));
    return 1;
  }

  auto plan = bind_mlp_plan(*model, *session, 0, rt->stream());
  if (!plan) {
    fail("bind: " + qw38::runtime::error_message(plan.error()));
    return 1;
  }
  expect(plan->swiglu.extent[0] == kFfn, "SwiGLU scratch is 17408 BF16");
  expect(plan->normalized.extent[0] == kHidden, "normalized scratch is 5120 BF16");
  expect(plan->h_mid.extent[0] == kHidden, "h_mid is 5120 FP32");
  expect(plan->next_h.extent[0] == kHidden, "next_h is 5120 FP32");
  expect(plan->h_mid.pointer == session->residual_h_mid().pointer,
         "MLP consumes session h_mid");
  expect(plan->next_h.pointer == session->residual_h().pointer,
         "MLP produces next-layer session h");
  expect(plan->stream == &rt->stream(), "ordered stream bound");

  auto residual = session->residual_h_mid();
  auto hbytes = qw38::activation::test::as_bytes(std::span<float const>(host.h_mid));
  auto up = qw38::cuda::copy_h2d(residual.pointer, hbytes.data(), hbytes.size(),
                                 rt->stream());
  expect(static_cast<bool>(up), "upload h_mid");
  if (!up) {
    return 1;
  }

  auto const nptr = plan->normalized.pointer;
  auto const sptr = plan->swiglu.pointer;
  auto const hptr = plan->h_mid.pointer;
  auto const next_ptr = plan->next_h.pointer;
  auto const mallocs = qw38::cuda::malloc_count();

  auto st = execute_decode_mlp(*plan);
  if (!st) {
    fail("execute1: " + qw38::runtime::error_message(st.error()));
    return 1;
  }
  expect(qw38::cuda::malloc_count() == mallocs, "first call allocates nothing");
  expect(plan->normalized.pointer == nptr && plan->swiglu.pointer == sptr &&
             plan->h_mid.pointer == hptr && plan->next_h.pointer == next_ptr,
         "scratch/residual addresses stable after first call");

  std::vector<float> source1(kHidden);
  std::vector<float> got1(kHidden);
  auto ds1 = qw38::cuda::copy_d2h(source1.data(), plan->h_mid.pointer,
                                 kHidden * sizeof(float), rt->stream());
  auto d1 = qw38::cuda::copy_d2h(got1.data(), plan->next_h.pointer,
                                kHidden * sizeof(float), rt->stream());
  expect(static_cast<bool>(ds1) && static_cast<bool>(d1),
         "download source/destination 1");
  auto sync1 = rt->stream().sync();
  expect(static_cast<bool>(sync1), "sync 1");
  expect_fp32_close(got1, cpu1->residual, "call1 residual", kResAbs, kResRel);
  expect_fp32_close(source1, host.h_mid, "call1 h_mid preserved", 0.0f, 0.0f);

  st = execute_decode_mlp(*plan);
  if (!st) {
    fail("execute2: " + qw38::runtime::error_message(st.error()));
    return 1;
  }
  expect(qw38::cuda::malloc_count() == mallocs, "second call allocates nothing");
  expect(plan->normalized.pointer == nptr && plan->swiglu.pointer == sptr &&
             plan->h_mid.pointer == hptr && plan->next_h.pointer == next_ptr,
         "scratch reuse across consecutive MLP calls");

  std::vector<float> source2(kHidden);
  std::vector<float> got2(kHidden);
  auto ds2 = qw38::cuda::copy_d2h(source2.data(), plan->h_mid.pointer,
                                 kHidden * sizeof(float), rt->stream());
  auto d2 = qw38::cuda::copy_d2h(got2.data(), plan->next_h.pointer,
                                kHidden * sizeof(float), rt->stream());
  expect(static_cast<bool>(ds2) && static_cast<bool>(d2),
         "download source/destination 2");
  auto sync2 = rt->stream().sync();
  expect(static_cast<bool>(sync2), "sync 2");
  expect_fp32_close(got2, cpu1->residual, "call2 residual", kResAbs, kResRel);
  expect_fp32_close(source2, host.h_mid, "call2 h_mid preserved", 0.0f, 0.0f);
  expect_fp32_close(got2, got1, "repeated execution is stable", 0.0f, 0.0f);

  if (g_failures != 0) {
    std::cerr << g_failures << " mlp integration failures\n";
    return 1;
  }
  std::cout << "mlp integration ok\n";
  return 0;
}
