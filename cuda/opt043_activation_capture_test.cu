#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "full_scheduler.h"
#include "mixer.h"
#include "model.h"
#include "scheduler.h"
#include "weights.h"

namespace {

constexpr std::size_t kCapacity = 131072;
constexpr char kPrefix[] = "QW38_OPT043_CAPTURE_RESULT=";

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s\n", status.message().c_str());
  return 1;
}

std::vector<std::size_t> formula_tokens(std::size_t count) {
  std::vector<std::size_t> tokens(count);
  for (std::size_t index = 0; index < count; ++index) {
    tokens[index] = (42 + index * 997) % qw38::internal::kVocabularySize;
  }
  return tokens;
}

std::vector<std::size_t> load_token_file(const char* path) {
  std::ifstream input(path);
  std::vector<std::size_t> tokens;
  std::size_t value = 0;
  while (input >> value) tokens.push_back(value);
  return tokens;
}

std::uint16_t fp32_to_bf16_bits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<std::uint16_t>(bits >> 16U);
}

bool write_f32(const std::string& path, const float* values, std::size_t count) {
  FILE* file = std::fopen(path.c_str(), "wb");
  if (file == nullptr) return false;
  const bool ok = std::fwrite(values, sizeof(float), count, file) == count;
  std::fclose(file);
  return ok;
}

bool write_bf16_from_f32(const std::string& path, const float* values,
                         std::size_t count) {
  std::vector<std::uint16_t> bits(count);
  for (std::size_t index = 0; index < count; ++index) {
    bits[index] = fp32_to_bf16_bits(values[index]);
  }
  FILE* file = std::fopen(path.c_str(), "wb");
  if (file == nullptr) return false;
  const bool ok =
      std::fwrite(bits.data(), sizeof(std::uint16_t), count, file) == count;
  std::fclose(file);
  return ok;
}

bool dump_vector(const std::string& dir, const std::string& stem,
                 const float* values, std::size_t count) {
  return write_f32(dir + "/" + stem + ".f32", values, count) &&
         write_bf16_from_f32(dir + "/" + stem + ".bf16", values, count);
}

void print_slot(const qw38::cuda::ActivationCaptureSlot& slot, bool last) {
  std::printf(
      "{\"layer\":%zu,\"layer_kind\":\"%s\",\"mixer_captured\":%s,"
      "\"ffn_captured\":%s,\"mixer_dtype\":\"%s\",\"ffn_dtype\":\"%s\","
      "\"mixer_shape\":[%zu,%zu],\"ffn_shape\":[%zu,%zu],"
      "\"mixer_sha256\":\"%s\",\"ffn_sha256\":\"%s\","
      "\"mixer_prefix\":[",
      slot.layer, slot.layer_kind, slot.mixer_captured ? "true" : "false",
      slot.ffn_captured ? "true" : "false", slot.mixer_dtype, slot.ffn_dtype,
      slot.mixer_shape[0], slot.mixer_shape[1], slot.ffn_shape[0],
      slot.ffn_shape[1], slot.mixer_sha256, slot.ffn_sha256);
  for (std::size_t index = 0; index < slot.mixer_prefix.size(); ++index) {
    if (index != 0) std::printf(",");
    std::printf("%.9g", static_cast<double>(slot.mixer_prefix[index]));
  }
  std::printf("],\"ffn_prefix\":[");
  for (std::size_t index = 0; index < slot.ffn_prefix.size(); ++index) {
    if (index != 0) std::printf(",");
    std::printf("%.9g", static_cast<double>(slot.ffn_prefix[index]));
  }
  std::printf("]}%s", last ? "" : ",");
}

}  // namespace

int main(int argc, char** argv) {
  const char* model_path = nullptr;
  const char* stage = nullptr;
  const char* token_file = nullptr;
  const char* dump_dir = nullptr;
  for (int index = 1; index < argc; ++index) {
    if (std::strcmp(argv[index], "--dump-dir") == 0 && index + 1 < argc) {
      dump_dir = argv[++index];
    } else if (model_path == nullptr) {
      model_path = argv[index];
    } else if (stage == nullptr) {
      stage = argv[index];
    } else if (token_file == nullptr) {
      token_file = argv[index];
    } else {
      std::fprintf(stderr,
                   "usage: %s MODEL.gguf STAGE [TOKEN_FILE] [--dump-dir DIR]\n"
                   "STAGE is d128, d2048, real_text, or t<position>\n",
                   argv[0]);
      return 2;
    }
  }
  if (model_path == nullptr || stage == nullptr) {
    std::fprintf(stderr,
                 "usage: %s MODEL.gguf STAGE [TOKEN_FILE] [--dump-dir DIR]\n",
                 argv[0]);
    return 2;
  }
  std::vector<std::size_t> tokens;
  if (std::strcmp(stage, "d128") == 0) {
    tokens = formula_tokens(129);
  } else if (std::strcmp(stage, "d2048") == 0) {
    tokens = formula_tokens(2049);
  } else if (stage[0] == 't' && stage[1] >= '0' && stage[1] <= '9') {
    const unsigned long position = std::strtoul(stage + 1, nullptr, 10);
    tokens = formula_tokens(position + 1);
  } else if (std::strcmp(stage, "real_text") == 0) {
    if (token_file == nullptr) {
      std::fprintf(stderr, "real_text requires TOKEN_FILE\n");
      return 2;
    }
    tokens = load_token_file(token_file);
    if (tokens.size() < 2) {
      std::fprintf(stderr, "TOKEN_FILE must contain at least two tokens\n");
      return 2;
    }
  } else {
    std::fprintf(stderr, "STAGE must be d128, d2048, real_text, or t<position>\n");
    return 2;
  }

  qw38::internal::ModelInfo info;
  qw38::Status status = qw38::internal::inspect_gguf(model_path, &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  qw38::internal::MappedFile mapping;
  if (status.is_ok()) status = mapping.open(model_path);
  qw38::internal::ModelWeights weights;
  if (status.is_ok()) {
    status = qw38::internal::bind_model_weights(info, mapping, &weights);
  }
  qw38::cuda::ResidentModel model;
  if (status.is_ok()) {
    status = model.upload(weights, mapping.data(), mapping.size());
  }
  qw38::cuda::SchedulerSession session;
  if (status.is_ok()) status = session.create(kCapacity);
  qw38::cuda::SchedulerWorkspace workspace;
  if (status.is_ok()) status = workspace.create(kCapacity);
  if (!status.is_ok()) return fail_status(status);

  const std::size_t prefix = tokens.size() - 1;
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  qw38::cuda::SyncResult result{};
  status = qw38::cuda::sync_tokens(
      model, tokens.data(), prefix, &session, &workspace, logits.data(),
      logits.size(), hidden.data(), hidden.size(), &result, nullptr, nullptr);
  if (!status.is_ok()) return fail_status(status);

  qw38::cuda::ActivationCapture capture;
  capture.stage = stage;
  capture.position = prefix;
  std::vector<std::vector<float>> down_store(capture.slots.size());
  std::vector<std::vector<float>> mix_store(capture.slots.size());
  for (std::size_t index = 0; index < capture.slots.size(); ++index) {
    capture.slots[index].layer = capture.layers[index];
    if (dump_dir != nullptr) {
      down_store[index].assign(qw38::internal::kFfnWidth, 0.0F);
      mix_store[index].assign(qw38::internal::kGdnValueWidth, 0.0F);
      capture.slots[index].down_full = down_store[index].data();
      capture.slots[index].mix_full = mix_store[index].data();
    }
  }
  float elapsed_ms = 0.0F;
  qw38::cuda::DecodeAttribution attribution;
  attribution.capture = &capture;
  status = qw38::cuda::execute_token(
      model, tokens[prefix], &session, &workspace, logits.data(), logits.size(),
      hidden.data(), hidden.size(), &elapsed_ms, nullptr, nullptr,
      qw38::cuda::PointwisePath::kUnfused, nullptr, &attribution);
  if (!status.is_ok()) return fail_status(status);

  for (const auto& slot : capture.slots) {
    if (!slot.mixer_captured || !slot.ffn_captured ||
        slot.mixer_sha256[0] == '\0' || slot.ffn_sha256[0] == '\0' ||
        std::strcmp(slot.mixer_sha256, capture.final_norm_sha256) == 0) {
      std::fprintf(stderr,
                   "OPT-043 capture substituted final hidden or missed a layer\n");
      return 1;
    }
  }
  if (!capture.final_norm_captured || !capture.output_captured ||
      capture.position != prefix) {
    std::fprintf(stderr, "OPT-043 capture missed final norm/output\n");
    return 1;
  }
  if (dump_dir != nullptr) {
    if (mkdir(dump_dir, 0755) != 0 && errno != EEXIST) {
      std::fprintf(stderr, "cannot create dump dir %s\n", dump_dir);
      return 1;
    }
    if (!dump_vector(dump_dir, "final_norm", capture.final_norm.data(),
                     capture.final_norm.size())) {
      std::fprintf(stderr, "cannot dump final_norm\n");
      return 1;
    }
    for (std::size_t index = 0; index < capture.slots.size(); ++index) {
      const auto& slot = capture.slots[index];
      const std::string prefix =
          std::string("L") + std::to_string(slot.layer);
      if (!dump_vector(dump_dir, prefix + "_mixer_preprojection",
                       slot.mixer_preprojection.data(),
                       slot.mixer_preprojection.size()) ||
          !dump_vector(dump_dir, prefix + "_ffn_preprojection",
                       slot.ffn_preprojection.data(),
                       slot.ffn_preprojection.size())) {
        std::fprintf(stderr, "cannot dump preprojection for layer %zu\n",
                     slot.layer);
        return 1;
      }
      if (slot.down_captured && slot.down_full != nullptr &&
          !dump_vector(dump_dir, prefix + "_swiglu_down_input", slot.down_full,
                       qw38::internal::kFfnWidth)) {
        std::fprintf(stderr, "cannot dump down input for layer %zu\n",
                     slot.layer);
        return 1;
      }
      if (slot.mix_captured && slot.mix_full != nullptr &&
          !dump_vector(dump_dir, prefix + "_gdn_or_attn_mix_6144", slot.mix_full,
                       qw38::internal::kGdnValueWidth)) {
        std::fprintf(stderr, "cannot dump mix input for layer %zu\n",
                     slot.layer);
        return 1;
      }
    }
    std::printf("dump_dir=%s position=%zu\n", dump_dir, capture.position);
  }

  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-043\",\"status\":\"measured\","
      "\"mode\":\"activation_capture\",\"stage\":\"%s\",\"position\":%zu,"
      "\"token_count\":%zu,\"unfused\":true,\"graphs_created\":false,"
      "\"final_norm_sha256\":\"%s\",\"output_sha256\":\"%s\","
      "\"output_count\":%zu,\"final_norm_prefix\":[",
      kPrefix, stage, capture.position, tokens.size(), capture.final_norm_sha256,
      capture.output_sha256, capture.output_count);
  for (std::size_t index = 0; index < capture.final_norm_prefix.size();
       ++index) {
    if (index != 0) std::printf(",");
    std::printf("%.9g", static_cast<double>(capture.final_norm_prefix[index]));
  }
  std::printf("],\"slots\":[");
  for (std::size_t index = 0; index < capture.slots.size(); ++index) {
    print_slot(capture.slots[index], index + 1 == capture.slots.size());
  }
  std::printf("]}\n");
  std::printf("status=passed\n");
  return 0;
}
