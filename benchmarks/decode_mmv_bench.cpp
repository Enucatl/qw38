#include "cuda/buffer.hpp"
#include "cuda/decode_mmv.hpp"
#include "cuda/error.hpp"
#include "cuda/event.hpp"
#include "cuda/stream.hpp"

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <iostream>
#include <span>
#include <string>
#include <utility>
#include <vector>

using qw38::cuda::decode_code_bytes;
using qw38::cuda::decode_scale_bytes;
using qw38::cuda::DecodeDtype;
using qw38::cuda::DecodeEpilogue;
using qw38::cuda::DecodeMmvDesc;
using qw38::cuda::DecodeMmvPairedDesc;
using qw38::cuda::DecodeMmvRangeDesc;
using qw38::cuda::DeviceBuffer;
using qw38::cuda::elapsed_ms;
using qw38::cuda::Event;
using qw38::cuda::kDecodeLayoutBf16DenseTileV0;
using qw38::cuda::kDecodeLayoutQ4G64V0;
using qw38::cuda::kDecodeLayoutQ4KCandidateV2;
using qw38::cuda::kDecodeLayoutQ8G32CandidateV1;
using qw38::cuda::kDecodeLayoutQ8G32V0;
using qw38::cuda::kDecodeQuantizerNone;
using qw38::cuda::kDecodeQuantizerQ4G64V0;
using qw38::cuda::kDecodeQuantizerQ4KCandidateV2;
using qw38::cuda::kDecodeQuantizerQ8G32CandidateV1;
using qw38::cuda::kDecodeQuantizerQ8G32V0;
using qw38::cuda::kDecodeThreads;
using qw38::cuda::kDecodeTileK;
using qw38::cuda::kDecodeWarps;
using qw38::cuda::launch_decode_ab_bf16;
using qw38::cuda::launch_decode_mmv;
using qw38::cuda::launch_decode_mmv_paired;
using qw38::cuda::launch_decode_mmv_ranges;
using qw38::cuda::Stream;
using qw38::cuda::zero;

namespace {

struct Case {
  char const* name;
  std::uint16_t layout;
  std::uint16_t quantizer;
  std::uint32_t n;
  std::uint32_t k;
  DecodeEpilogue epi;
  bool grouped_ab;
};

struct GroupCase {
  char const* name;
  std::uint16_t layout;
  std::uint16_t quantizer;
  std::array<std::uint32_t, 3> widths;
  std::uint32_t k;
  std::size_t count;
  bool swiglu;
};

std::expected<float, qw38::cuda::Error> time_launch(Stream const& stream,
                                                    DecodeMmvDesc const& d,
                                                    int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) {
    auto st = launch_decode_mmv(d, stream);
    if (!st) {
      return std::unexpected(st.error());
    }
  }
  auto t0 = Event::create_timing();
  auto t1 = Event::create_timing();
  if (!t0 || !t1) {
    return std::unexpected(t0 ? t1.error() : t0.error());
  }
  auto r0 = t0->record(stream);
  if (!r0) {
    return std::unexpected(r0.error());
  }
  for (int i = 0; i < iters; ++i) {
    auto st = launch_decode_mmv(d, stream);
    if (!st) {
      return std::unexpected(st.error());
    }
  }
  auto r1 = t1->record(stream);
  if (!r1) {
    return std::unexpected(r1.error());
  }
  auto s1 = t1->sync();
  if (!s1) {
    return std::unexpected(s1.error());
  }
  auto ms = elapsed_ms(*t0, *t1);
  if (!ms) {
    return std::unexpected(ms.error());
  }
  return *ms / static_cast<float>(iters);
}

}  // namespace

int main() {
  cudaDeviceProp prop{};
  if (auto st = qw38::cuda::check(cudaGetDeviceProperties(&prop, 0),
                                  "cudaGetDeviceProperties");
      !st) {
    std::cerr << qw38::cuda::error_message(st.error()) << '\n';
    return 1;
  }
  auto stream = Stream::create();
  if (!stream) {
    std::cerr << qw38::cuda::error_message(stream.error()) << '\n';
    return 1;
  }

  std::cout << "qw38_bench_decode_mmv\n";
  std::cout << "device=" << prop.name << " sm_" << prop.major << prop.minor
            << '\n';
  std::cout << "geometry warps/block=" << kDecodeWarps
            << " threads=" << kDecodeThreads << " k_tile=" << kDecodeTileK
            << " max_k=" << qw38::cuda::kDecodeMaxK << '\n';

  Case cases[] = {
      {"q4-8x256", kDecodeLayoutQ4G64V0, kDecodeQuantizerQ4G64V0, 8, 256,
       DecodeEpilogue::StoreBf16, false},
      {"q4-mlp-down-5120x17408", kDecodeLayoutQ4G64V0, kDecodeQuantizerQ4G64V0,
       5120, 17408, DecodeEpilogue::ResidualAddFp32, false},
      {"q4-mlp-gate-17408x5120", kDecodeLayoutQ4G64V0, kDecodeQuantizerQ4G64V0,
       17408, 5120, DecodeEpilogue::StoreBf16, false},
      {"q4k-mlp-down-5120x17408", kDecodeLayoutQ4KCandidateV2,
       kDecodeQuantizerQ4KCandidateV2, 5120, 17408,
       DecodeEpilogue::ResidualAddFp32, false},
      {"q4k-mlp-gate-17408x5120", kDecodeLayoutQ4KCandidateV2,
       kDecodeQuantizerQ4KCandidateV2, 17408, 5120, DecodeEpilogue::StoreBf16,
       false},
      {"q8-candidate-gdn-qkv-10240x5120", kDecodeLayoutQ8G32CandidateV1,
       kDecodeQuantizerQ8G32CandidateV1, 10240, 5120, DecodeEpilogue::StoreBf16,
       false},
      {"q8-candidate-gdn-out-5120x6144", kDecodeLayoutQ8G32CandidateV1,
       kDecodeQuantizerQ8G32CandidateV1, 5120, 6144,
       DecodeEpilogue::ResidualAddFp32, false},
      {"q8-candidate-attn-q-12288x5120", kDecodeLayoutQ8G32CandidateV1,
       kDecodeQuantizerQ8G32CandidateV1, 12288, 5120, DecodeEpilogue::StoreBf16,
       false},
      {"q8-candidate-attn-out-5120x6144", kDecodeLayoutQ8G32CandidateV1,
       kDecodeQuantizerQ8G32CandidateV1, 5120, 6144,
       DecodeEpilogue::ResidualAddFp32, false},
      {"q8-candidate-head-248320x5120", kDecodeLayoutQ8G32CandidateV1,
       kDecodeQuantizerQ8G32CandidateV1, 248320, 5120,
       DecodeEpilogue::StoreFp32, false},
      {"q8-head-248320x5120", kDecodeLayoutQ8G32V0, kDecodeQuantizerQ8G32V0,
       248320, 5120, DecodeEpilogue::StoreFp32, false},
      {"bf16-ab-48x5120", kDecodeLayoutBf16DenseTileV0, kDecodeQuantizerNone,
       48, 5120, DecodeEpilogue::StoreFp32, true},
  };

  for (auto const& c : cases) {
    auto const pn = c.n;
    auto const pk = c.k;
    auto const code_n = decode_code_bytes(c.layout, pn, pk);
    auto const scale_n = decode_scale_bytes(c.layout, pn, pk);
    auto codes = DeviceBuffer::allocate(code_n);
    if (!codes) {
      std::cerr << c.name
                << " codes: " << qw38::cuda::error_message(codes.error())
                << '\n';
      return 1;
    }
    auto zc = zero(*codes, *stream);
    if (!zc) {
      std::cerr << qw38::cuda::error_message(zc.error()) << '\n';
      return 1;
    }
    std::expected<DeviceBuffer, qw38::cuda::Error> scales;
    if (scale_n != 0) {
      scales = DeviceBuffer::allocate(scale_n);
      if (!scales) {
        std::cerr << c.name << " scales\n";
        return 1;
      }
      auto zs = zero(*scales, *stream);
      if (!zs) {
        return 1;
      }
    }
    auto input = DeviceBuffer::allocate(static_cast<std::uint64_t>(c.k) * 2);
    if (!input) {
      std::cerr << c.name << " input\n";
      return 1;
    }
    auto zi = zero(*input, *stream);
    if (!zi) {
      return 1;
    }

    DecodeMmvDesc d;
    d.layout = c.layout;
    d.quantizer = c.quantizer;
    d.n = c.n;
    d.k = c.k;
    d.padded_n = pn;
    d.padded_k = pk;
    DecodeDtype const weight_dtype =
        (c.layout == kDecodeLayoutQ4G64V0 ||
         c.layout == kDecodeLayoutQ4KCandidateV2)
            ? DecodeDtype::Q4
            : ((c.layout == kDecodeLayoutQ8G32V0 ||
                c.layout == kDecodeLayoutQ8G32CandidateV1)
                   ? DecodeDtype::Q8
                   : DecodeDtype::Bf16);
    d.codes = qw38::cuda::decode_matrix_view(
        codes->data(), weight_dtype, c.layout, c.n, c.k, pn, pk, code_n, 16);
    if (scale_n != 0) {
      auto const units_per_row = static_cast<std::uint32_t>(scale_n / pn / 2u);
      d.scales = qw38::cuda::decode_matrix_view(
          scales->data(), DecodeDtype::Fp16, c.layout, pn, units_per_row, pn,
          units_per_row, scale_n, 2);
    }
    d.input = qw38::cuda::decode_vector_view(
        input->data(), DecodeDtype::Bf16, qw38::cuda::kDecodeLayoutBf16VectorV0,
        c.k, static_cast<std::uint64_t>(c.k) * 2u, 2, false);
    d.epilogue = c.epi;

    std::expected<DeviceBuffer, qw38::cuda::Error> out;
    std::expected<DeviceBuffer, qw38::cuda::Error> residual;
    std::expected<DeviceBuffer, qw38::cuda::Error> out_b;
    std::expected<DeviceBuffer, qw38::cuda::Error> codes_b;
    if (c.epi == DecodeEpilogue::ResidualAddFp32) {
      out = DeviceBuffer::allocate(static_cast<std::uint64_t>(c.n) * 4);
      residual = DeviceBuffer::allocate(static_cast<std::uint64_t>(c.n) * 4);
      if (!out || !residual) {
        return 1;
      }
      if (auto zr = zero(*residual, *stream); !zr) {
        return 1;
      }
      d.residual = qw38::cuda::decode_vector_view(
          residual->data(), DecodeDtype::Fp32,
          qw38::cuda::kDecodeLayoutFp32VectorV0, c.n,
          static_cast<std::uint64_t>(c.n) * 4u, 4, false);
      d.output = qw38::cuda::decode_vector_view(
          out->data(), DecodeDtype::Fp32, qw38::cuda::kDecodeLayoutFp32VectorV0,
          c.n, static_cast<std::uint64_t>(c.n) * 4u, 4, true);
    } else if (c.epi == DecodeEpilogue::StoreFp32) {
      out = DeviceBuffer::allocate(static_cast<std::uint64_t>(c.n) * 4);
      if (!out) {
        return 1;
      }
      d.output = qw38::cuda::decode_vector_view(
          out->data(), DecodeDtype::Fp32, qw38::cuda::kDecodeLayoutFp32VectorV0,
          c.n, static_cast<std::uint64_t>(c.n) * 4u, 4, true);
    } else {
      out = DeviceBuffer::allocate(static_cast<std::uint64_t>(c.n) * 2);
      if (!out) {
        return 1;
      }
      d.output = qw38::cuda::decode_vector_view(
          out->data(), DecodeDtype::Bf16, qw38::cuda::kDecodeLayoutBf16VectorV0,
          c.n, static_cast<std::uint64_t>(c.n) * 2u, 2, true);
    }

    float ms = 0.0f;
    int const iterations = 8;
    unsigned const grid = c.n / 8u;
    DecodeMmvPairedDesc paired;
    if (c.grouped_ab) {
      codes_b = DeviceBuffer::allocate(code_n);
      out_b = DeviceBuffer::allocate(static_cast<std::uint64_t>(c.n) * 4);
      if (!codes_b || !out_b) {
        return 1;
      }
      auto zb = zero(*codes_b, *stream);
      if (!zb) {
        return 1;
      }
      paired.a = d;
      paired.b = d;
      paired.b.codes.pointer = codes_b->data();
      paired.b.output = qw38::cuda::decode_vector_view(
          out_b->data(), DecodeDtype::Fp32,
          qw38::cuda::kDecodeLayoutFp32VectorV0, c.n,
          static_cast<std::uint64_t>(c.n) * 4u, 4, true);
      for (int i = 0; i < 2; ++i) {
        auto st = launch_decode_ab_bf16(paired, *stream);
        if (!st) {
          std::cerr << qw38::cuda::error_message(st.error()) << '\n';
          return 1;
        }
      }
      auto t0 = Event::create_timing();
      auto t1 = Event::create_timing();
      if (!t0 || !t1) {
        auto const& error = t0 ? t1.error() : t0.error();
        std::cerr << c.name << ' ' << qw38::cuda::error_message(error) << '\n';
        return 1;
      }
      auto r0 = t0->record(*stream);
      if (!r0) {
        std::cerr << c.name << ' ' << qw38::cuda::error_message(r0.error())
                  << '\n';
        return 1;
      }
      for (int i = 0; i < iterations; ++i) {
        auto st = launch_decode_ab_bf16(paired, *stream);
        if (!st) {
          std::cerr << c.name << ' ' << qw38::cuda::error_message(st.error())
                    << '\n';
          return 1;
        }
      }
      auto r1 = t1->record(*stream);
      if (!r1) {
        std::cerr << c.name << ' ' << qw38::cuda::error_message(r1.error())
                  << '\n';
        return 1;
      }
      auto s1 = t1->sync();
      if (!s1) {
        std::cerr << c.name << ' ' << qw38::cuda::error_message(s1.error())
                  << '\n';
        return 1;
      }
      auto elapsed = elapsed_ms(*t0, *t1);
      if (!elapsed) {
        std::cerr << qw38::cuda::error_message(elapsed.error()) << '\n';
        return 1;
      }
      ms = *elapsed / static_cast<float>(iterations);
    } else {
      auto timed = time_launch(*stream, d, 2, iterations);
      if (!timed) {
        std::cerr << c.name << ' ' << qw38::cuda::error_message(timed.error())
                  << '\n';
        return 1;
      }
      ms = *timed;
    }
    std::vector<std::byte> host_output(out->bytes());
    std::vector<std::byte> host_output_b(c.grouped_ab ? out_b->bytes() : 0);
    auto const host_start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
      auto st = c.grouped_ab ? launch_decode_ab_bf16(paired, *stream)
                             : launch_decode_mmv(d, *stream);
      if (!st) {
        std::cerr << c.name << ' ' << qw38::cuda::error_message(st.error())
                  << '\n';
        return 1;
      }
      if (auto synced = stream->sync(); !synced) {
        std::cerr << c.name << ' ' << qw38::cuda::error_message(synced.error())
                  << '\n';
        return 1;
      }
      if (auto copied = qw38::cuda::check(
              cudaMemcpy(host_output.data(), out->data(), out->bytes(),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy benchmark output");
          !copied) {
        std::cerr << c.name << ' ' << qw38::cuda::error_message(copied.error())
                  << '\n';
        return 1;
      }
      if (c.grouped_ab) {
        if (auto copied = qw38::cuda::check(
                cudaMemcpy(host_output_b.data(), out_b->data(), out_b->bytes(),
                           cudaMemcpyDeviceToHost),
                "cudaMemcpy benchmark paired output");
            !copied) {
          std::cerr << c.name << ' '
                    << qw38::cuda::error_message(copied.error()) << '\n';
          return 1;
        }
      }
    }
    auto const host_complete_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - host_start)
            .count() /
        iterations;
    std::uint64_t bytes =
        code_n + scale_n + static_cast<std::uint64_t>(c.k) * 2u;
    if (c.grouped_ab) {
      bytes += code_n;
    }
    std::cout << c.name << " n=" << c.n << " k=" << c.k << " layout=0x"
              << std::hex << c.layout << std::dec << " code_bytes=" << code_n
              << " scale_bytes=" << scale_n << " traffic_bytes=" << bytes
              << " grid=" << grid << " iterations=" << iterations
              << " kernel_launches=" << iterations
              << " device_to_device_copies=0" << " launch_path="
              << (c.grouped_ab ? "paired_bf16" : "single_isolated")
              << " epilogue=" << static_cast<int>(c.epi)
              << " input_scaling=none activation=bf16" << " kernel_ms=" << ms
              << " host_complete_ms=" << host_complete_ms
              << " host_readout_copies=" << iterations * (c.grouped_ab ? 2 : 1)
              << " readout_bytes="
              << out->bytes() + (c.grouped_ab ? out_b->bytes() : 0) << '\n';
  }

  GroupCase grouped[] = {
      {"q4k-mlp-gate-up-17408x5120-paired",
       kDecodeLayoutQ4KCandidateV2,
       kDecodeQuantizerQ4KCandidateV2,
       {17408, 17408, 0},
       5120,
       2,
       true},
      {"q8-candidate-gdn-qkv-z-ranges",
       kDecodeLayoutQ8G32CandidateV1,
       kDecodeQuantizerQ8G32CandidateV1,
       {10240, 6144, 0},
       5120,
       2,
       false},
      {"q8-candidate-attn-qg-k-v-ranges",
       kDecodeLayoutQ8G32CandidateV1,
       kDecodeQuantizerQ8G32CandidateV1,
       {12288, 1024, 1024},
       5120,
       3,
       false},
  };
  for (auto const& c : grouped) {
    auto input = DeviceBuffer::allocate(static_cast<std::uint64_t>(c.k) * 2u);
    if (!input || !zero(*input, *stream)) {
      std::cerr << c.name << " input allocation/zero failed\n";
      return 1;
    }
    std::array<DeviceBuffer, 3> codes;
    std::array<DeviceBuffer, 3> scales;
    std::array<DeviceBuffer, 3> outputs;
    std::array<DecodeMmvDesc, 3> descs{};
    std::uint64_t traffic_bytes = input->bytes();
    std::uint64_t readout_bytes = 0;
    unsigned grid = 0;
    for (std::size_t i = 0; i < c.count; ++i) {
      auto const n = c.widths[i];
      auto const code_bytes = decode_code_bytes(c.layout, n, c.k);
      auto const scale_bytes = decode_scale_bytes(c.layout, n, c.k);
      auto allocated_codes = DeviceBuffer::allocate(code_bytes);
      auto allocated_scales = DeviceBuffer::allocate(scale_bytes);
      if (!allocated_codes || !allocated_scales) {
        std::cerr << c.name << " weight allocation failed\n";
        return 1;
      }
      codes[i] = std::move(*allocated_codes);
      scales[i] = std::move(*allocated_scales);
      if (!zero(codes[i], *stream) || !zero(scales[i], *stream)) {
        std::cerr << c.name << " weight zero failed\n";
        return 1;
      }
      auto& d = descs[i];
      d.layout = c.layout;
      d.quantizer = c.quantizer;
      d.n = n;
      d.k = c.k;
      d.padded_n = n;
      d.padded_k = c.k;
      d.codes = qw38::cuda::decode_matrix_view(
          codes[i].data(), c.swiglu ? DecodeDtype::Q4 : DecodeDtype::Q8,
          c.layout, n, c.k, n, c.k, code_bytes, 16);
      auto const scale_units = static_cast<std::uint32_t>(scale_bytes / n / 2u);
      d.scales = qw38::cuda::decode_matrix_view(
          scales[i].data(), DecodeDtype::Fp16, c.layout, n, scale_units, n,
          scale_units, scale_bytes, 2);
      d.input = qw38::cuda::decode_vector_view(
          input->data(), DecodeDtype::Bf16,
          qw38::cuda::kDecodeLayoutBf16VectorV0, c.k, input->bytes(), 2, false);
      d.epilogue = c.swiglu ? DecodeEpilogue::SwigluStoreBf16
                            : DecodeEpilogue::StoreBf16;
      if (!c.swiglu || i == 0) {
        auto allocated_output =
            DeviceBuffer::allocate(static_cast<std::uint64_t>(n) * 2u);
        if (!allocated_output) {
          std::cerr << c.name << " output allocation failed\n";
          return 1;
        }
        outputs[i] = std::move(*allocated_output);
        d.output = qw38::cuda::decode_vector_view(
            outputs[i].data(), DecodeDtype::Bf16,
            qw38::cuda::kDecodeLayoutBf16VectorV0, n, outputs[i].bytes(), 2,
            true);
        readout_bytes += outputs[i].bytes();
      }
      traffic_bytes += code_bytes + scale_bytes;
      grid += n / 8u;
    }
    if (c.swiglu) grid /= 2u;
    auto launch = [&]() -> std::expected<void, qw38::cuda::Error> {
      if (c.swiglu) {
        return launch_decode_mmv_paired(DecodeMmvPairedDesc{descs[0], descs[1]},
                                        *stream);
      }
      return launch_decode_mmv_ranges(
          DecodeMmvRangeDesc{
              std::span<DecodeMmvDesc const>(descs.data(), c.count)},
          *stream);
    };
    for (int i = 0; i < 2; ++i) {
      if (auto st = launch(); !st) {
        std::cerr << c.name << ' ' << qw38::cuda::error_message(st.error())
                  << '\n';
        return 1;
      }
    }
    auto t0 = Event::create_timing();
    auto t1 = Event::create_timing();
    if (!t0 || !t1) return 1;
    if (auto st = t0->record(*stream); !st) return 1;
    int const iterations = 8;
    for (int i = 0; i < iterations; ++i) {
      if (auto st = launch(); !st) {
        std::cerr << c.name << ' ' << qw38::cuda::error_message(st.error())
                  << '\n';
        return 1;
      }
    }
    if (auto st = t1->record(*stream); !st) return 1;
    if (auto st = t1->sync(); !st) return 1;
    auto elapsed = elapsed_ms(*t0, *t1);
    if (!elapsed) return 1;
    std::vector<std::byte> host_output(readout_bytes);
    auto const host_start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
      if (auto st = launch(); !st) {
        std::cerr << c.name << ' ' << qw38::cuda::error_message(st.error())
                  << '\n';
        return 1;
      }
      if (auto st = stream->sync(); !st) return 1;
      std::size_t offset = 0;
      for (auto const& output : outputs) {
        if (output.empty()) continue;
        if (auto st = qw38::cuda::check(
                cudaMemcpy(host_output.data() + offset, output.data(),
                           output.bytes(), cudaMemcpyDeviceToHost),
                "cudaMemcpy grouped benchmark output");
            !st) {
          std::cerr << c.name << ' ' << qw38::cuda::error_message(st.error())
                    << '\n';
          return 1;
        }
        offset += output.bytes();
      }
    }
    auto const host_complete_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - host_start)
            .count() /
        iterations;
    std::cout << c.name << " n=";
    for (std::size_t i = 0; i < c.count; ++i) {
      if (i != 0) std::cout << ',';
      std::cout << c.widths[i];
    }
    std::cout << " k=" << c.k << " layout=0x" << std::hex << c.layout
              << std::dec << " traffic_bytes=" << traffic_bytes
              << " grid=" << grid << " iterations=" << iterations
              << " kernel_launches=" << iterations
              << " device_to_device_copies=0" << " launch_path="
              << (c.swiglu ? "paired_swiglu" : "grouped_ranges")
              << " epilogue=" << (c.swiglu ? "swiglu_store_bf16" : "store_bf16")
              << " input_scaling=none activation=bf16"
              << " kernel_ms=" << *elapsed / iterations
              << " host_complete_ms=" << host_complete_ms
              << " host_readout_copies="
              << iterations * (c.swiglu ? 1 : c.count)
              << " readout_bytes=" << readout_bytes << '\n';
  }
  return 0;
}
