#include "cuda/buffer.hpp"
#include "cuda/decode_mmv.hpp"
#include "cuda/error.hpp"
#include "cuda/event.hpp"
#include "cuda/stream.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <expected>
#include <iostream>
#include <string>
#include <vector>

using qw38::cuda::DecodeEpilogue;
using qw38::cuda::DecodeMmvDesc;
using qw38::cuda::DecodeMmvPairedDesc;
using qw38::cuda::DeviceBuffer;
using qw38::cuda::Event;
using qw38::cuda::Stream;
using qw38::cuda::decode_code_bytes;
using qw38::cuda::decode_scale_bytes;
using qw38::cuda::elapsed_ms;
using qw38::cuda::kDecodeLayoutBf16DenseTileV0;
using qw38::cuda::kDecodeLayoutQ4G64V0;
using qw38::cuda::kDecodeLayoutQ8G32V0;
using qw38::cuda::kDecodeQuantizerNone;
using qw38::cuda::kDecodeQuantizerQ4G64V0;
using qw38::cuda::kDecodeQuantizerQ8G32V0;
using qw38::cuda::kDecodeThreads;
using qw38::cuda::kDecodeTileK;
using qw38::cuda::kDecodeWarps;
using qw38::cuda::launch_decode_ab_bf16;
using qw38::cuda::launch_decode_mmv;
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

std::expected<float, qw38::cuda::Error> time_launch(Stream const& stream,
                                                    DecodeMmvDesc const& d, int warmup,
                                                    int iters) {
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
  std::cout << "device=" << prop.name << " sm_" << prop.major << prop.minor << '\n';
  std::cout << "geometry warps/block=" << kDecodeWarps
            << " threads=" << kDecodeThreads << " k_tile=" << kDecodeTileK
            << " max_k=" << qw38::cuda::kDecodeMaxK << '\n';

  Case cases[] = {
      {"q4-8x256", kDecodeLayoutQ4G64V0, kDecodeQuantizerQ4G64V0, 8, 256,
       DecodeEpilogue::StoreBf16, false},
      {"q4-mlp-down-5120x17408", kDecodeLayoutQ4G64V0, kDecodeQuantizerQ4G64V0, 5120,
       17408, DecodeEpilogue::ResidualAddFp32, false},
      {"q4-mlp-gate-17408x5120", kDecodeLayoutQ4G64V0, kDecodeQuantizerQ4G64V0, 17408,
       5120, DecodeEpilogue::StoreBf16, false},
      {"q8-head-248320x5120", kDecodeLayoutQ8G32V0, kDecodeQuantizerQ8G32V0, 248320,
       5120, DecodeEpilogue::StoreFp32, false},
      {"bf16-ab-48x5120", kDecodeLayoutBf16DenseTileV0, kDecodeQuantizerNone, 48, 5120,
       DecodeEpilogue::StoreFp32, true},
  };

  for (auto const& c : cases) {
    auto const pn = c.n;
    auto const pk = c.k;
    auto const code_n = decode_code_bytes(c.layout, pn, pk);
    auto const scale_n = decode_scale_bytes(c.layout, pn, pk);
    auto codes = DeviceBuffer::allocate(code_n);
    if (!codes) {
      std::cerr << c.name << " codes: " << qw38::cuda::error_message(codes.error())
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
    d.codes = codes->as_bytes();
    d.codes_bytes = code_n;
    if (scale_n != 0) {
      d.scales = scales->as_bytes();
      d.scales_bytes = scale_n;
    }
    d.input = static_cast<std::uint16_t const*>(input->data());
    d.epilogue = c.epi;

    std::expected<DeviceBuffer, qw38::cuda::Error> out;
    std::expected<DeviceBuffer, qw38::cuda::Error> out_b;
    std::expected<DeviceBuffer, qw38::cuda::Error> codes_b;
    if (c.epi == DecodeEpilogue::ResidualAddFp32) {
      out = DeviceBuffer::allocate(static_cast<std::uint64_t>(c.n) * 4);
      if (!out) {
        return 1;
      }
      auto zo = zero(*out, *stream);
      if (!zo) {
        return 1;
      }
      d.residual = static_cast<float*>(out->data());
    } else if (c.epi == DecodeEpilogue::StoreFp32) {
      out = DeviceBuffer::allocate(static_cast<std::uint64_t>(c.n) * 4);
      if (!out) {
        return 1;
      }
      d.output = out->data();
    } else {
      out = DeviceBuffer::allocate(static_cast<std::uint64_t>(c.n) * 2);
      if (!out) {
        return 1;
      }
      d.output = out->data();
    }

    float ms = 0.0f;
    int const launches = 8;
    unsigned const grid = c.n / 8u;
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
      DecodeMmvPairedDesc p;
      p.a = d;
      p.codes_b = codes_b->as_bytes();
      p.codes_b_bytes = code_n;
      p.output_b = out_b->data();
      for (int i = 0; i < 2; ++i) {
        auto st = launch_decode_ab_bf16(p, *stream);
        if (!st) {
          std::cerr << qw38::cuda::error_message(st.error()) << '\n';
          return 1;
        }
      }
      auto t0 = Event::create_timing();
      auto t1 = Event::create_timing();
      if (!t0 || !t1) {
        return 1;
      }
      (void)t0->record(*stream);
      for (int i = 0; i < launches; ++i) {
        auto st = launch_decode_ab_bf16(p, *stream);
        if (!st) {
          return 1;
        }
      }
      (void)t1->record(*stream);
      (void)t1->sync();
      auto elapsed = elapsed_ms(*t0, *t1);
      if (!elapsed) {
        std::cerr << qw38::cuda::error_message(elapsed.error()) << '\n';
        return 1;
      }
      ms = *elapsed / static_cast<float>(launches);
    } else {
      auto timed = time_launch(*stream, d, 2, launches);
      if (!timed) {
        std::cerr << c.name << ' ' << qw38::cuda::error_message(timed.error()) << '\n';
        return 1;
      }
      ms = *timed;
    }
    std::uint64_t bytes = code_n + scale_n + static_cast<std::uint64_t>(c.k) * 2u;
    if (c.grouped_ab) {
      bytes += code_n;
    }
    std::cout << c.name << " n=" << c.n << " k=" << c.k << " layout=0x" << std::hex
              << c.layout << std::dec << " code_bytes=" << code_n
              << " scale_bytes=" << scale_n << " traffic_bytes=" << bytes
              << " grid=" << grid << " launches=" << launches
              << " ms=" << ms << '\n';
  }
  return 0;
}
