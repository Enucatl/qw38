#include "runtime/language_model.hpp"
#include "runtime/runtime.hpp"

#include "cuda/activation.hpp"
#include "cuda/copy.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace qw38::runtime {

struct LanguageLayerPlanTestAccess {
  static std::expected<void, Error> write_attention(
      std::filesystem::path const& directory, std::string const& kind,
      std::uint64_t position, void const* source, std::size_t bytes,
      cuda::Stream const& stream) {
    std::vector<char> host(bytes);
    if (auto st = cuda::copy_d2h(host.data(), source, bytes, stream); !st) {
      return std::unexpected(from_cuda(st.error()));
    }
    if (auto st = stream.sync(); !st) return std::unexpected(from_cuda(st.error()));
    auto const path = directory / ("attention." + kind + ".pos-" +
                                   std::to_string(position) + ".bin");
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(host.data(), static_cast<std::streamsize>(bytes));
    if (!output) {
      return std::unexpected(make_error(ErrorCode::Internal, "trace",
                                        "could not write attention tensor"));
    }
    return {};
  }

  template <typename Callback>
  static std::expected<void, Error> execute_attention_trace(
      LanguageLayerPlan const& layer, std::uint64_t position,
      std::filesystem::path const& directory, Callback&& after_mixer) {
    auto const& prep = layer.attention_.prep;
    auto const& core = layer.attention_.core;
    auto save = [&](std::string const& kind, void const* source,
                    std::size_t bytes) {
      return write_attention(directory, kind, position, source, bytes, *prep.stream);
    };
    if (auto st = save("input", prep.residual.pointer, kHidden * sizeof(float)); !st) return st;
    if (auto st = execute_decode_attention_prep(prep, position); !st) return st;
    if (auto st = save("normalized", prep.normalized.pointer, kHidden * 2u); !st) return st;
    if (auto st = save("qg", prep.scratch.qg.pointer, 24u * 512u * 2u); !st) return st;
    if (auto st = save("k_raw", prep.scratch.k.pointer, 4u * 256u * 2u); !st) return st;
    if (auto st = save("v_raw", prep.scratch.v.pointer, 4u * 256u * 2u); !st) return st;
    if (auto st = save("q", prep.scratch.q.pointer, 24u * 256u * 2u); !st) return st;
    if (auto st = save("g", prep.scratch.g.pointer, 24u * 256u * 2u); !st) return st;
    std::array<std::uint16_t, 2u * 4u * 256u> current_kv{};
    auto const* kv = static_cast<std::uint16_t const*>(prep.kv.pointer);
    std::size_t const head_stride = prep.kv_capacity * 256u;
    std::size_t const layer_offset = prep.attn_layer * 2u * 4u * head_stride;
    for (std::size_t component = 0; component < 2; ++component) {
      for (std::size_t head = 0; head < 4; ++head) {
        auto const* source = kv + layer_offset +
                             (component * 4u + head) * head_stride + position * 256u;
        auto* target = current_kv.data() + (component * 4u + head) * 256u;
        if (auto st = cuda::copy_d2h(target, source, 256u * 2u, *prep.stream); !st) {
          return std::unexpected(from_cuda(st.error()));
        }
      }
    }
    if (auto st = prep.stream->sync(); !st) return std::unexpected(from_cuda(st.error()));
    auto const kv_path = directory / ("attention.kv.pos-" +
                                      std::to_string(position) + ".bin");
    std::ofstream kv_output(kv_path, std::ios::binary | std::ios::trunc);
    kv_output.write(reinterpret_cast<char const*>(current_kv.data()),
                    static_cast<std::streamsize>(sizeof(current_kv)));
    if (!kv_output) {
      return std::unexpected(make_error(ErrorCode::Internal, "trace",
                                        "could not write attention KV"));
    }
    auto mixer = execute_attention_core(core);
    if (!mixer) return std::unexpected(mixer.error());
    if (auto st = save("gated", core.y.pointer, 24u * 256u * 2u); !st) return st;
    if (auto st = save("output", core.residual_out.pointer, kHidden * sizeof(float)); !st) return st;
    if (auto st = after_mixer(core.residual_out.pointer); !st) return st;
    return execute_decode_mlp(layer.mlp_);
  }

  template <typename Callback>
  static std::expected<void, Error> execute_split(LanguageLayerPlan const& layer,
                                                  std::uint64_t position,
                                                  Callback&& after_mixer) {
    auto mixer = execute_decode_attention(layer.attention_, position);
    if (!mixer) return std::unexpected(mixer.error());
    if (auto st = after_mixer(layer.attention_.core.residual_out.pointer); !st) return st;
    return execute_decode_mlp(layer.mlp_);
  }
};

struct LanguageModelPlanTestAccess {
  static std::expected<void, Error> trace_token(LanguageModelPlan& plan,
                                                std::uint32_t token,
                                                std::uint64_t position,
                                                std::filesystem::path const& directory) {
    if (position != plan.state_->token_position() ||
        !plan.state_->layers_at(position)) {
      return std::unexpected(make_error(ErrorCode::InvalidPopulatedLength,
                                        "position", "trace is not at a token boundary"));
    }
    if (auto st = cuda::launch_embed_gather(plan.embedding_, kVocab, token,
                                            plan.residual_, *plan.stream_); !st) {
      return std::unexpected(from_cuda(st.error()));
    }
    bool const checkpoint = position == 0 || position == 63 ||
                            position == 255 || position == 383 || position == 384;
    std::array<float, kHidden> host{};
    auto write_residual = [&](std::string const& kind, std::uint32_t layer,
                              void const* source)
        -> std::expected<void, Error> {
      if (auto st = cuda::copy_d2h(host.data(), source,
                                   kResidualBytesPerToken, *plan.stream_); !st) {
        return std::unexpected(from_cuda(st.error()));
      }
      if (auto st = plan.stream_->sync(); !st) {
        return std::unexpected(from_cuda(st.error()));
      }
      auto const path = directory / (kind + ".pos-" + std::to_string(position) +
                                     ".layer-" + std::to_string(layer) + ".f32");
      std::ofstream output(path, std::ios::binary | std::ios::trunc);
      output.write(reinterpret_cast<char const*>(host.data()),
                   static_cast<std::streamsize>(kResidualBytesPerToken));
      if (!output) {
        return std::unexpected(make_error(ErrorCode::Internal, "trace",
                                          "could not write residual"));
      }
      return {};
    };
    for (auto const& layer : plan.layers_) {
      bool const split = layer.layer() == 3 || layer.layer() == 7 ||
                         layer.layer() == 31 || layer.layer() == 63;
      if (layer.layer() == 3) {
        if (auto st = LanguageLayerPlanTestAccess::execute_attention_trace(
                layer, position, directory, [&](void const* source) -> std::expected<void, Error> {
                  if (checkpoint) return write_residual("mixer", layer.layer(), source);
                  return {};
                }); !st) {
          return std::unexpected(st.error());
        }
      } else if (checkpoint && split) {
        if (auto st = LanguageLayerPlanTestAccess::execute_split(
                layer, position, [&](void const* source) {
                  return write_residual("mixer", layer.layer(), source);
                }); !st) {
          return std::unexpected(st.error());
        }
      } else {
        auto result = execute_decode_language_layer(layer, position);
        if (!result) return std::unexpected(result.error());
      }
      if (!checkpoint) continue;
      if (auto st = write_residual("residual", layer.layer(), plan.residual_); !st) return st;
    }
    if (!plan.state_->layers_at(position + 1)) {
      return std::unexpected(make_error(ErrorCode::Internal, "trace",
                                        "a layer did not advance its state"));
    }
    plan.state_->commit_token();
    return {};
  }
};

}  // namespace qw38::runtime

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: qw38-task022-trace ARTIFACT TOKENS.u32le OUTPUT_DIR\n";
    return 2;
  }
  using namespace qw38::runtime;
  std::ifstream input(argv[2], std::ios::binary);
  if (!input) return 1;
  std::vector<std::uint32_t> tokens(385);
  for (auto& token : tokens) {
    std::array<unsigned char, 4> bytes{};
    input.read(reinterpret_cast<char*>(bytes.data()), 4);
    if (!input) return 1;
    token = static_cast<std::uint32_t>(bytes[0]) |
            (static_cast<std::uint32_t>(bytes[1]) << 8) |
            (static_cast<std::uint32_t>(bytes[2]) << 16) |
            (static_cast<std::uint32_t>(bytes[3]) << 24);
    if (token >= kVocab) return 1;
  }
  std::filesystem::path directory(argv[3]);
  std::filesystem::create_directories(directory);
  auto runtime = Runtime::create();
  if (!runtime) {
    std::cerr << error_message(runtime.error()) << '\n';
    return 1;
  }
  auto model = runtime->load(argv[1]);
  if (!model) {
    std::cerr << error_message(model.error()) << '\n';
    return 1;
  }
  auto session = runtime->create_session(*model, tokens.size());
  if (!session) {
    std::cerr << error_message(session.error()) << '\n';
    return 1;
  }
  auto plan = LanguageModelPlan::bind(*model, *session, runtime->stream());
  if (!plan) {
    std::cerr << error_message(plan.error()) << '\n';
    return 1;
  }
  for (std::size_t position = 0; position < tokens.size(); ++position) {
    if (auto st = LanguageModelPlanTestAccess::trace_token(
            *plan, tokens[position], position, directory); !st) {
      std::cerr << error_message(st.error()) << '\n';
      return 1;
    }
    if (position != 0 && position != 63 && position != 255 &&
        position != 383 && position != 384) continue;
    auto snapshot = session->save();
    if (!snapshot || snapshot->gdn_s.size() < kGdnSBytesPerLayer) return 1;
    auto const path = directory / ("gdn-state.pos-" + std::to_string(position) + ".f32");
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<char const*>(snapshot->gdn_s.data()),
                 static_cast<std::streamsize>(kGdnSBytesPerLayer));
    if (!output) return 1;
    std::cout << "traced_position=" << position << '\n' << std::flush;
  }
  return 0;
}
