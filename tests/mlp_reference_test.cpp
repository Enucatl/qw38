#include "mlp_support.hpp"

#include "compiler/quantization/quantizer.hpp"

#include <iostream>
#include <string>

using qw38::mlp::test::DeviceMlp;
using qw38::mlp::test::HostMlp;
using qw38::mlp::test::bf16_matrix;
using qw38::mlp::test::bf16_vec;
using qw38::mlp::test::bind_views;
using qw38::mlp::test::bytes_to_u16;
using qw38::mlp::test::decoded_weights;
using qw38::mlp::test::download_vec;
using qw38::mlp::test::expect;
using qw38::mlp::test::expect_bf16_close;
using qw38::mlp::test::expect_fp32_close;
using qw38::mlp::test::fail;
using qw38::mlp::test::fill_logical_pattern;
using qw38::mlp::test::g_failures;
using qw38::mlp::test::kNormAbs;
using qw38::mlp::test::kResAbs;
using qw38::mlp::test::kResRel;
using qw38::mlp::test::kSwigluAbs;
using qw38::mlp::test::kSwigluRel;
using qw38::mlp::test::make_logical;
using qw38::mlp::test::pack_q4_from_logical;
using qw38::mlp::test::residual_vec;
using qw38::mlp::test::upload_host_mlp;
using qw38::cuda::Stream;
using qw38::format::LogicalQuantizerId;
using qw38::format::PhysicalLayoutId;
using qw38::reference::decode_mlp_reference;
using qw38::reference::kDefaultRmsEps;
using qw38::reference::kFfn;
using qw38::reference::kHidden;
using qw38::runtime::bind_mlp_plan;
using qw38::runtime::execute_decode_mlp;

namespace {

bool run_against_reference(HostMlp& host, Stream const& stream,
                           std::string_view tag) {
  auto cpu = decode_mlp_reference(host.h_mid, host.gamma, kDefaultRmsEps,
                                  host.w_gate, host.w_up, host.w_down);
  if (!cpu) {
    fail(std::string(tag) + " cpu: " +
         qw38::reference::error_message(cpu.error()));
    return false;
  }
  DeviceMlp dev;
  if (!upload_host_mlp(host, dev, stream)) {
    return false;
  }
  auto views = bind_views(dev);
  auto plan = bind_mlp_plan(views, stream);
  if (!plan) {
    fail(std::string(tag) + " bind: " +
         qw38::runtime::error_message(plan.error()));
    return false;
  }
  auto st = execute_decode_mlp(*plan);
  if (!st) {
    fail(std::string(tag) + " execute: " +
         qw38::runtime::error_message(st.error()));
    return false;
  }
  auto norm = download_vec<std::uint16_t>(dev.normalized, kHidden, stream);
  auto sw = download_vec<std::uint16_t>(dev.swiglu, kFfn, stream);
  auto res = download_vec<float>(dev.residual, kHidden, stream);
  if (!norm || !sw || !res) {
    fail(std::string(tag) + " download");
    return false;
  }
  expect_bf16_close(*norm, cpu->normalized, std::string(tag) + " norm", kNormAbs,
                    0.0f);
  expect_bf16_close(*sw, cpu->swiglu, std::string(tag) + " swiglu", kSwigluAbs,
                    kSwigluRel);
  expect_fp32_close(*res, cpu->residual, std::string(tag) + " residual", kResAbs,
                    kResRel);
  return true;
}

void packed_q4(Stream const& stream) {
  HostMlp host;
  auto lg = make_logical(LogicalQuantizerId::Q4G64V0, kFfn, kHidden, 1, 0x3C00);
  auto lu = make_logical(LogicalQuantizerId::Q4G64V0, kFfn, kHidden, -2, 0x3C00);
  auto ld = make_logical(LogicalQuantizerId::Q4G64V0, kHidden, kFfn, 3, 0x3C00);
  fill_logical_pattern(lg, 2);
  fill_logical_pattern(lu, 5);
  fill_logical_pattern(ld, 8);
  if (!pack_q4_from_logical(lg, host.gate, "q4 gate") ||
      !pack_q4_from_logical(lu, host.up, "q4 up") ||
      !pack_q4_from_logical(ld, host.down, "q4 down")) {
    return;
  }
  auto wg = qw38::compiler::dequantize_to_bf16(lg);
  auto wu = qw38::compiler::dequantize_to_bf16(lu);
  auto wd = qw38::compiler::dequantize_to_bf16(ld);
  if (!wg || !wu || !wd) {
    fail("q4 dequant");
    return;
  }
  host.w_gate = std::move(*wg);
  host.w_up = std::move(*wu);
  host.w_down = std::move(*wd);
  host.gamma = bf16_vec(kHidden, 0.08f);
  host.h_mid = residual_vec(kHidden, 0.35f);
  (void)run_against_reference(host, stream, "q4-mlp");
}

void bf16_control(Stream const& stream) {
  auto src_g = bf16_matrix(kFfn, kHidden, 0.45f, 1);
  auto src_u = bf16_matrix(kFfn, kHidden, 0.55f, 4);
  auto src_d = bf16_matrix(kHidden, kFfn, 0.40f, 9);
  auto pg = qw38::format::pack_bf16_dense_tile_v0(src_g, kFfn, kHidden);
  auto pu = qw38::format::pack_bf16_dense_tile_v0(src_u, kFfn, kHidden);
  auto pd = qw38::format::pack_bf16_dense_tile_v0(src_d, kHidden, kFfn);
  if (!pg || !pu || !pd) {
    fail("bf16 pack");
    return;
  }
  HostMlp host;
  host.gate = std::move(*pg);
  host.up = std::move(*pu);
  host.down = std::move(*pd);
  host.w_gate = bytes_to_u16(src_g);
  host.w_up = bytes_to_u16(src_u);
  host.w_down = bytes_to_u16(src_d);
  host.gamma = bf16_vec(kHidden, 0.12f);
  host.h_mid = residual_vec(kHidden, 0.22f);
  (void)run_against_reference(host, stream, "bf16-control-mlp");
}

void cpu_shape_error() {
  std::vector<float> h(8, 0.0f);
  std::vector<std::uint16_t> g(kHidden, 0);
  std::vector<std::uint16_t> w(1, 0);
  auto bad = decode_mlp_reference(h, g, kDefaultRmsEps, w, w, w);
  expect(!bad && bad.error().code == qw38::reference::ErrorCode::InvalidShape,
         "cpu mlp rejects short residual");
}

}  // namespace

int main() {
  cpu_shape_error();
  auto stream = Stream::create();
  if (!stream) {
    fail("stream");
    return 1;
  }
  packed_q4(*stream);
  bf16_control(*stream);
  if (g_failures != 0) {
    std::cerr << g_failures << " mlp reference failures\n";
    return 1;
  }
  std::cout << "mlp reference ok\n";
  return 0;
}
