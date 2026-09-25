#include "decode_mmv_support.hpp"
#include "compiler/checkpoint.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {
using namespace qw38::decode_mmv::test;
using qw38::compiler::quantize_bf16;
using qw38::format::pack_cuda_v0;

template<class T> std::vector<T> read(std::filesystem::path const& path, std::size_t count) {
  std::vector<T> data(count);
  std::ifstream input(path, std::ios::binary);
  if (!input.read(reinterpret_cast<char*>(data.data()), count * sizeof(T))) {
    throw std::runtime_error("cannot read " + path.string());
  }
  return data;
}

std::vector<float> contractions(PackedMatrix const& matrix,
                                std::vector<std::uint16_t> const& input,
                                unsigned count, Stream const& stream) {
  auto codes = upload_vec(matrix.codes, stream);
  auto x = upload_vec(input, stream);
  auto output = DeviceBuffer::allocate(matrix.logical_n * count * sizeof(float));
  std::expected<DeviceBuffer, qw38::cuda::Error> scales;
  if (!matrix.scales.empty()) scales = upload_vec(matrix.scales, stream);
  if (!codes || !x || !output || (!matrix.scales.empty() && !scales)) {
    throw std::runtime_error("projection upload failed");
  }
  auto d = desc_from_packed(matrix, DecodeEpilogue::StoreFp32);
  d.codes.pointer = codes->data();
  if (!matrix.scales.empty()) d.scales.pointer = scales->data();
  for (unsigned row = 0; row < count; ++row) {
    bind_input(d, static_cast<std::uint16_t*>(x->data()) + row * matrix.logical_k);
    bind_output(d, static_cast<float*>(output->data()) + row * matrix.logical_n);
    auto st = qw38::cuda::launch_decode_mmv(d, stream);
    if (!st) throw std::runtime_error(qw38::cuda::error_message(st.error()));
  }
  auto values = download_vec<float>(*output, matrix.logical_n * count, stream);
  if (!values) throw std::runtime_error("projection download failed");
  return *values;
}

void projection_data(std::span<std::byte const> weights,
                std::vector<std::uint16_t> const& inputs, std::string const& family,
                unsigned n, unsigned k, unsigned count, Stream const& stream, int layer = -1) {
  std::vector<float> reference;
  {
    auto packed = qw38::format::pack_bf16_dense_tile_v0(weights, n, k);
    if (!packed) throw std::runtime_error("BF16 pack failed");
    reference = contractions(*packed, inputs, count, stream);
  }
  for (auto const& [quantizer, layout] : {
           std::pair{LogicalQuantizerId::Q4G64CandidateV1, PhysicalLayoutId::CudaQ4G64CandidateV1},
           std::pair{LogicalQuantizerId::Q4KCandidateV2, PhysicalLayoutId::CudaQ4KCandidateV2}}) {
    auto logical = quantize_bf16(quantizer, n, k, weights);
    if (!logical) throw std::runtime_error(qw38::compiler::error_message(logical.error()));
    auto packed = pack_cuda_v0(quantizer, layout, *logical);
    if (!packed) throw std::runtime_error("quantized pack failed");
    auto actual = contractions(*packed, inputs, count, stream);
    unsigned const phases = layer < 0 ? 1 : 2;
    for (unsigned phase = 0; phase < phases; ++phase) {
    unsigned const first = phase == 0 ? 0 : count - 1;
    unsigned const rows = phases == 1 ? count : (phase == 0 ? count - 1 : 1);
    double square_error = 0, square_reference = 0, maximum_error = 0;
    for (std::size_t i = std::size_t{first} * n; i < std::size_t{first + rows} * n; ++i) {
      double const error = static_cast<double>(actual[i]) - reference[i];
      square_error += error * error;
      square_reference += static_cast<double>(reference[i]) * reference[i];
      maximum_error = std::max(maximum_error, std::abs(error));
    }
    std::cout << std::setprecision(12) << "{\"family\":\"" << family
              << "\",\"quantizer\":\"" << (quantizer == LogicalQuantizerId::Q4KCandidateV2 ? "q4_k_candidate_v2" : "q4_g64_candidate_v1")
              << "\",\"input_first\":0,\"input_count\":" << rows
              << ",\"n\":" << n << ",\"k\":" << k
              << ",\"relative_l2\":" << std::sqrt(square_error / square_reference)
              << ",\"rmse\":" << std::sqrt(square_error / (std::size_t{rows} * n))
              << ",\"max_abs_error\":" << maximum_error
              << ",\"reference\":\"original_bf16_weights_quartz_fp32_contraction\"";
    if (layer >= 0) std::cout << ",\"layer\":" << layer << ",\"phase\":\""
        << (phase == 0 ? "prefill" : "decode") << "\",\"input_source\":\"task020_saved_f32_rounded_bf16\"";
    std::cout << "}" << std::endl;
    }
  }
}
void projection(std::filesystem::path const& root, std::string const& family,
                unsigned n, unsigned k, unsigned count, Stream const& stream) {
  auto weights = read<std::byte>(root / family / "b.bf16le", std::size_t{n} * k * 2);
  auto inputs = read<std::uint16_t>(root / family / "a.bf16le", std::size_t{count} * k);
  projection_data(weights, inputs, family, n, k, count, stream);
}

void traces(Stream const& stream) {
  auto checkpoint = qw38::compiler::open_checkpoint(".cache/authorities/qwen3.8-27b-transformers");
  if (!checkpoint) throw std::runtime_error(qw38::compiler::error_message(checkpoint.error()));
  for (int layer : {0, 31, 63}) {
    for (std::string family : {"gate", "up", "down"}) {
      unsigned const n = family == "down" ? 5120 : 17408;
      unsigned const k = family == "down" ? 17408 : 5120;
      std::string const name = "model.language_model.layers." + std::to_string(layer)
          + ".mlp." + family + "_proj.weight";
      auto found = std::find_if(checkpoint->classified.included.begin(), checkpoint->classified.included.end(),
          [&](auto const& tensor) { return tensor.expected.name == name; });
      if (found == checkpoint->classified.included.end()) throw std::runtime_error("missing weight " + name);
      auto shard = qw38::compiler::MappedShard::open(checkpoint->shards.at(found->source.shard).path);
      if (!shard) throw std::runtime_error(qw38::compiler::error_message(shard.error()));
      auto weights = shard->tensor_bytes(found->source);
      if (!weights) throw std::runtime_error(qw38::compiler::error_message(weights.error()));
      if (weights->size() != std::size_t{n} * k * 2) throw std::runtime_error("wrong weight dimensions");
      std::vector<std::uint16_t> inputs;
      for (std::string phase : {"prefill", "decode"}) {
        std::string const base = ".cache/task020/trace/blk." + std::to_string(layer)
            + ".ffn_" + (family == "down" ? "down" : "gate") + ".weight." + phase + ".bin";
        // Trace sidecars contain literal \t separators; fixed geometry is checked
        // against the byte length before converting original saved FP32 inputs.
        unsigned const rows = phase == "prefill" ? 384 : 1;
        if (std::filesystem::file_size(base) != std::size_t{rows} * k * sizeof(float))
          throw std::runtime_error("unexpected trace dimensions " + base);
        auto values = read<float>(base, std::size_t{rows} * k);
        for (float value : values) {
          if (!std::isfinite(value)) throw std::runtime_error("nonfinite trace input");
          inputs.push_back(qw38::format::fp32_to_bf16_rne(value));
        }
      }
      projection_data(*weights, inputs, "mlp_" + family, n, k, 385, stream, layer);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    auto stream = Stream::create();
    if (!stream) throw std::runtime_error("cannot create CUDA stream");
    if (argc > 1 && std::string_view(argv[1]) == "--traces") { traces(*stream); return 0; }
    std::filesystem::path const root = argc > 1 ? argv[1] : ".cache/task019/inputs/real-matrix";
    unsigned const count = argc > 2 ? std::stoul(argv[2]) : 32;
    if (count == 0 || count > 1024) throw std::runtime_error("input count must be 1..1024");
    projection(root, "mlp_gate", 17408, 5120, count, *stream);
    projection(root, "mlp_up", 17408, 5120, count, *stream);
    projection(root, "mlp_down", 5120, 17408, count, *stream);
    return 0;
  } catch (std::exception const& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
