#include "runtime/language_model.hpp"
#include "runtime/runtime.hpp"

#include "cuda/activation.hpp"
#include "cuda/copy.hpp"
#include "cuda/decode_mmv.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

bool parse_token(std::string_view text, std::uint32_t& value) {
  auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc{} && end == text.data() + text.size();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "usage: qw38-bf16-diagnostic ARTIFACT OUTPUT_PREFIX TOKEN...\n";
    return 2;
  }
  using namespace qw38;
  using namespace qw38::runtime;
  std::vector<std::uint32_t> tokens;
  for (int i = 3; i < argc; ++i) {
    std::uint32_t token{};
    if (!parse_token(argv[i], token) || token >= kVocab) return 2;
    tokens.push_back(token);
  }
  auto fail = [](Error const& error) {
    std::cerr << error_message(error) << '\n';
    return 1;
  };
  auto artifact = format::Artifact::open(argv[1]);
  if (!artifact) {
    std::cerr << format::error_message(artifact.error()) << '\n';
    return 1;
  }
  if (auto st = validate_primary_language_graph(artifact->schema()); !st) {
    return fail(st.error());
  }
  auto const* head_record = [&]() -> format::TensorRecord const* {
    for (auto const& record : artifact->schema().tensors) {
      if (record.logical_name == "lm_head.weight") return &record;
    }
    return nullptr;
  }();
  if (head_record == nullptr || head_record->storage != format::StorageClass::Bf16 ||
      head_record->layout != format::PhysicalLayoutId::CudaBf16DenseTileV0) {
    return fail(make_error(ErrorCode::InvalidArgument, "artifact",
                           "BF16 diagnostic requires an identity artifact"));
  }
  auto runtime = Runtime::create();
  if (!runtime) return fail(runtime.error());
  auto input = runtime->upload_diagnostic(*artifact, DiagnosticWeights::Input);
  if (!input) return fail(input.error());
  std::optional<Model> input_model(std::move(*input));
  auto session = runtime->create_session(*input_model, tokens.size());
  if (!session) return fail(session.error());
  auto* state = detail::SessionPlanAccess::execution_state(*session);
  std::uint64_t peak_model_bytes = input_model->device_bytes();
  std::vector<float> host_logits(kVocab);
  std::vector<float> host_residual(kHidden);
  for (std::uint64_t position = 0; position < tokens.size(); ++position) {
    std::ofstream residual_file(std::string(argv[2]) + "-residuals-" +
                                    std::to_string(position) + ".f32",
                                std::ios::binary | std::ios::trunc);
    if (!residual_file) return 1;
    if (position > 0) {
      auto uploaded = runtime->upload_diagnostic(*artifact, DiagnosticWeights::Input);
      if (!uploaded) return fail(uploaded.error());
      input_model.emplace(std::move(*uploaded));
    }
    auto embedding = input_model->payload("model.language_model.embed_tokens.weight");
    if (!embedding) return fail(embedding.error());
    if (auto st = cuda::launch_embed_gather(
            static_cast<std::uint16_t const*>(embedding->pointer), kVocab,
            tokens[position], static_cast<float*>(session->residual_h().pointer),
            runtime->stream()); !st) {
      return fail(from_cuda(st.error()));
    }
    if (auto st = runtime->stream().sync(); !st) return fail(from_cuda(st.error()));
    peak_model_bytes = std::max(peak_model_bytes, input_model->device_bytes());
    input_model.reset();

    for (std::uint32_t layer = 0; layer < kLanguageLayers; ++layer) {
      auto weights = runtime->upload_diagnostic(*artifact,
                                                 DiagnosticWeights::Layer, layer);
      if (!weights) return fail(weights.error());
      peak_model_bytes = std::max(peak_model_bytes, weights->device_bytes());
      auto plan = LanguageLayerPlan::bind(*weights, *session, layer,
                                          runtime->stream());
      if (!plan) return fail(plan.error());
      auto result = execute_decode_language_layer(*plan, position);
      if (!result) return fail(result.error());
      // The next layer upload frees these weights; wait for its MLP launch.
      if (auto st = cuda::copy_d2h(host_residual.data(),
                                   session->residual_h().pointer,
                                   kResidualBytesPerToken, runtime->stream()); !st) {
        return fail(from_cuda(st.error()));
      }
      if (auto st = runtime->stream().sync(); !st) return fail(from_cuda(st.error()));
      residual_file.write(reinterpret_cast<char const*>(host_residual.data()),
                          static_cast<std::streamsize>(kResidualBytesPerToken));
      if (!residual_file) return 1;
    }
    if (!state->layers_at(position + 1)) {
      return fail(make_error(ErrorCode::Internal, "position",
                             "BF16 diagnostic did not advance every layer"));
    }
    auto output = runtime->upload_diagnostic(*artifact, DiagnosticWeights::Output);
    if (!output) return fail(output.error());
    peak_model_bytes = std::max(peak_model_bytes, output->device_bytes());
    auto norm = output->payload("model.language_model.norm.weight");
    auto head = output->payload("lm_head.weight");
    auto normalized = session->scratch(format::ScratchKind::NormalizedHidden);
    auto logits = session->scratch(format::ScratchKind::Logits);
    if (!norm) return fail(norm.error());
    if (!head) return fail(head.error());
    if (!normalized) return fail(normalized.error());
    if (!logits) return fail(logits.error());
    auto* norm_out = static_cast<std::uint16_t*>(normalized->region[0].tensor.pointer);
    auto* logits_out = static_cast<float*>(logits->region[0].tensor.pointer);
    if (auto st = cuda::launch_hidden_rms(
            static_cast<float*>(session->residual_h().pointer),
            static_cast<std::uint16_t const*>(norm->pointer), kMlpRmsEps,
            1, norm_out, runtime->stream()); !st) {
      return fail(from_cuda(st.error()));
    }
    cuda::DecodeMmvDesc desc;
    desc.layout = cuda::kDecodeLayoutBf16DenseTileV0;
    desc.quantizer = cuda::kDecodeQuantizerNone;
    desc.n = kVocab;
    desc.k = kHidden;
    desc.padded_n = static_cast<std::uint32_t>(cuda::decode_pad_n(kVocab));
    desc.padded_k = cuda::decode_pad_k(kHidden);
    desc.codes = cuda::decode_matrix_view(
        const_cast<void*>(head->pointer), cuda::DecodeDtype::Bf16, desc.layout,
        desc.n, desc.k, desc.padded_n, desc.padded_k,
        cuda::decode_code_bytes(desc.layout, desc.padded_n, desc.padded_k), 16);
    desc.input = cuda::decode_vector_view(
        norm_out, cuda::DecodeDtype::Bf16,
        cuda::kDecodeLayoutBf16VectorV0, kHidden,
        static_cast<std::uint64_t>(kHidden) * 2u, 2, false);
    desc.output = cuda::decode_vector_view(
        logits_out, cuda::DecodeDtype::Fp32,
        cuda::kDecodeLayoutFp32VectorV0, kVocab,
        kLogitsBytesPerToken, 4, true);
    desc.epilogue = cuda::DecodeEpilogue::StoreFp32;
    if (auto st = cuda::launch_decode_mmv(desc, runtime->stream()); !st) {
      return fail(from_cuda(st.error()));
    }
    if (auto st = cuda::copy_d2h(host_logits.data(), logits_out,
                                 kLogitsBytesPerToken, runtime->stream()); !st) {
      return fail(from_cuda(st.error()));
    }
    if (auto st = runtime->stream().sync(); !st) return fail(from_cuda(st.error()));
    if (!std::ranges::all_of(host_logits, [](float x) { return std::isfinite(x); })) {
      return fail(make_error(ErrorCode::Internal, "logits", "non-finite BF16 logits"));
    }
    state->commit_token();
    auto snapshot = session->save();
    if (!snapshot) return fail(snapshot.error());
    auto write_state = [&](std::string const& kind, std::uint32_t layer,
                           std::byte const* data, std::size_t bytes) {
      std::ofstream state_file(std::string(argv[2]) + "-" + kind + "-" +
                                   std::to_string(position) + "-" +
                                   std::to_string(layer) + ".bin",
                               std::ios::binary | std::ios::trunc);
      state_file.write(reinterpret_cast<char const*>(data),
                       static_cast<std::streamsize>(bytes));
      return static_cast<bool>(state_file);
    };
    for (auto layer : {0u, 28u, 60u}) {
      auto index = gdn_state_index(layer);
      if (!index) return fail(index.error());
      if (!write_state("recurrent", layer,
                       snapshot->gdn_s.data() + *index * kGdnSBytesPerLayer,
                       kGdnSBytesPerLayer)) return 1;
      constexpr std::size_t conv_layer_bytes =
          kConvTaps * kConvChannels * format::kBf16Size;
      if (!write_state("conv", layer,
                       snapshot->conv_history.data() + *index * conv_layer_bytes,
                       conv_layer_bytes)) return 1;
    }
    constexpr std::size_t kv_layer_bytes =
        kKvComponents * kKvHeads * kHeadDim * format::kBf16Size;
    auto attention_index = attn_state_index(3);
    if (!attention_index) return fail(attention_index.error());
    if (!write_state("kv", 3,
                     snapshot->kv.data() + *attention_index * tokens.size() * kv_layer_bytes,
                     tokens.size() * kv_layer_bytes)) return 1;
    auto path = std::filesystem::path(std::string(argv[2]) + "-" +
                                      std::to_string(position) + ".f32");
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<char const*>(host_logits.data()),
               static_cast<std::streamsize>(kLogitsBytesPerToken));
    if (!file) return 1;
    auto const best = std::max_element(host_logits.begin(), host_logits.end());
    std::cout << "position=" << position << " argmax="
              << std::distance(host_logits.begin(), best) << '\n';
  }
  std::cout << "peak_diagnostic_model_bytes=" << peak_model_bytes
            << " session_persistent_bytes=" << session->persistent_bytes()
            << " session_scratch_bytes=" << session->arena_plan().total_bytes
            << " session_residual_bytes="
            << 2u * kArenaTokenCapacity * kResidualBytesPerToken
            << " retained_MTP=disabled\n";
  return 0;
}
