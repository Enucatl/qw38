#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "full_scheduler.h"
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
  if (argc < 3 || argc > 4) {
    std::fprintf(stderr,
                 "usage: %s MODEL.gguf STAGE [TOKEN_FILE]\n"
                 "STAGE is d128, d2048, or real_text\n",
                 argv[0]);
    return 2;
  }
  const char* stage = argv[2];
  std::vector<std::size_t> tokens;
  if (std::strcmp(stage, "d128") == 0) {
    tokens = formula_tokens(129);
  } else if (std::strcmp(stage, "d2048") == 0) {
    tokens = formula_tokens(2049);
  } else if (std::strcmp(stage, "real_text") == 0) {
    if (argc != 4) {
      std::fprintf(stderr, "real_text requires TOKEN_FILE\n");
      return 2;
    }
    tokens = load_token_file(argv[3]);
    if (tokens.size() < 2) {
      std::fprintf(stderr, "TOKEN_FILE must contain at least two tokens\n");
      return 2;
    }
  } else {
    std::fprintf(stderr, "STAGE must be d128, d2048, or real_text\n");
    return 2;
  }

  qw38::internal::ModelInfo info;
  qw38::Status status = qw38::internal::inspect_gguf(argv[1], &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  qw38::internal::MappedFile mapping;
  if (status.is_ok()) status = mapping.open(argv[1]);
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
  for (std::size_t index = 0; index < capture.layers.size(); ++index) {
    capture.slots[index].layer = capture.layers[index];
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
