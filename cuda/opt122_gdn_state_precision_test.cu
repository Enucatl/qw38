#include "engine_attribution.h"
#include "execution_graph_path.cuh"
#include "full_scheduler.h"
#include "opt110_engine_hook.cuh"
#include "opt122_gdn_state_precision.cuh"
#include "test_tier.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "model.h"
#include "scheduler.h"
#include "weights.h"

namespace {

constexpr char kPrefix[] = "QW38_OPT122_GDN_STATE_PRECISION_RESULT=";
constexpr char kCounts[] = "QW38_OPT122_NATIVE_COUNTS=";
constexpr std::size_t kMinCapacity = 4096;
constexpr int kRecurrenceWarmups = 5;
constexpr int kRecurrenceIters = 20;

struct Options final {
  const char* workload = "inventory";
  const char* execution_graphs = "ffn_only";
  std::size_t prefix = 128;
  std::size_t capacity = 0;
};

const char* json_bool(bool value) { return value ? "true" : "false"; }

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s: %s\n", qw38::status_code_name(status.code()),
               status.message().c_str());
  return 1;
}

int usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s [--workload reference|inventory] "
               "[--execution-graphs ffn_only] [--prefix N] [--capacity N] "
               "MODEL\n",
               argv0);
  return 2;
}

bool parse_size(const char* text, std::size_t* value) {
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(text, &end, 10);
  if (end == text || *end != '\0') return false;
  *value = static_cast<std::size_t>(parsed);
  return true;
}

int parse_args(int argc, char** argv, Options* options) {
  int model_index = -1;
  for (int index = 1; index < argc; ++index) {
    const char* arg = argv[index];
    if (std::strcmp(arg, "--workload") == 0 && index + 1 < argc) {
      options->workload = argv[++index];
    } else if (std::strcmp(arg, "--execution-graphs") == 0 && index + 1 < argc) {
      options->execution_graphs = argv[++index];
    } else if (std::strcmp(arg, "--prefix") == 0 && index + 1 < argc) {
      if (!parse_size(argv[++index], &options->prefix)) return usage(argv[0]);
    } else if (std::strcmp(arg, "--capacity") == 0 && index + 1 < argc) {
      if (!parse_size(argv[++index], &options->capacity)) return usage(argv[0]);
    } else if (arg[0] == '-') {
      return usage(argv[0]);
    } else {
      model_index = index;
    }
  }
  if (model_index < 0) return usage(argv[0]);
  if (!qw38::cuda::legal_execution_graph_path(options->execution_graphs)) {
    std::fprintf(stderr, "invalid --execution-graphs %s\n",
                 options->execution_graphs);
    return 2;
  }
  return model_index;
}

void print_counts(std::size_t warmups, std::size_t samples,
                  std::size_t candidates, bool keep) {
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-122\",\"observed_warmups\":%zu,"
      "\"observed_samples\":%zu,\"observed_candidates\":%zu,"
      "\"observed_shapes\":1,\"observed_tier\":\"screen\",\"keep\":%s}\n",
      kCounts, warmups, samples, candidates, json_bool(keep));
}

void fill_tokens(std::vector<std::size_t>* tokens) {
  for (std::size_t index = 0; index < tokens->size(); ++index) {
    (*tokens)[index] = (42 + index * 997) % qw38::internal::kVocabularySize;
  }
}

std::size_t session_capacity(std::size_t prefix, std::size_t outputs,
                             std::size_t requested) {
  if (requested != 0) return requested;
  return std::max(prefix + outputs + 32, kMinCapacity);
}

qw38::Status load_model(const char* path, qw38::internal::MappedFile* mapping,
                        qw38::cuda::ResidentModel* model) {
  qw38::internal::ModelInfo info;
  qw38::Status status = qw38::internal::inspect_gguf(path, &info);
  if (status.is_ok()) status = qw38::internal::validate_qwen38_contract(&info);
  if (status.is_ok()) status = mapping->open(path);
  qw38::internal::ModelWeights weights;
  if (status.is_ok()) {
    status = qw38::internal::bind_model_weights(info, *mapping, &weights);
  }
  if (status.is_ok()) {
    status = model->upload(weights, mapping->data(), mapping->size());
  }
  return status;
}

qw38::Status create_graphs(const qw38::cuda::ResidentModel& model,
                           qw38::cuda::SchedulerWorkspace* workspace,
                           qw38::cuda::SchedulerGraphs* graphs) {
  qw38::cuda::ExecutionGraphPathScope scope(
      qw38::cuda::kLegalExecutionGraphFfnOnly);
  return graphs->create(model, workspace);
}

__global__ void opt122_recurrence_fp32(const float* source, float* candidate,
                                       float* output, const float* query,
                                       const float* key, const float* value,
                                       float decay, float beta) {
  const std::uint32_t value_head = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (value_head >= qw38::cuda::kGdnStateValueHeads ||
      lane >= qw38::cuda::kGdnStateHeadWidth)
    return;
  const std::size_t head_base =
      static_cast<std::size_t>(value_head) * qw38::cuda::kGdnStateHeadWidth *
      qw38::cuda::kGdnStateHeadWidth;
  float prediction = 0.0F;
  for (std::uint32_t key_lane = 0; key_lane < qw38::cuda::kGdnStateHeadWidth;
       ++key_lane) {
    const std::size_t index =
        head_base + static_cast<std::size_t>(key_lane) *
                        qw38::cuda::kGdnStateHeadWidth +
        lane;
    prediction = __fadd_rn(prediction, __fmul_rn(source[index], decay));
  }
  const float delta = __fmul_rn(value[lane] - prediction, beta);
  float result = 0.0F;
  for (std::uint32_t key_lane = 0; key_lane < qw38::cuda::kGdnStateHeadWidth;
       ++key_lane) {
    const std::size_t index =
        head_base + static_cast<std::size_t>(key_lane) *
                        qw38::cuda::kGdnStateHeadWidth +
        lane;
    const float updated =
        __fadd_rn(__fmul_rn(source[index], decay),
                  __fmul_rn(key[key_lane], delta));
    candidate[index] = updated;
    result = __fadd_rn(result, __fmul_rn(query[key_lane], updated));
  }
  output[static_cast<std::size_t>(value_head) * qw38::cuda::kGdnStateHeadWidth +
         lane] = result;
}

__global__ void opt122_recurrence_bf16(const __nv_bfloat16* source,
                                       __nv_bfloat16* candidate, float* output,
                                       const float* query, const float* key,
                                       const float* value, float decay,
                                       float beta) {
  const std::uint32_t value_head = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (value_head >= qw38::cuda::kGdnStateValueHeads ||
      lane >= qw38::cuda::kGdnStateHeadWidth)
    return;
  const std::size_t head_base =
      static_cast<std::size_t>(value_head) * qw38::cuda::kGdnStateHeadWidth *
      qw38::cuda::kGdnStateHeadWidth;
  float prediction = 0.0F;
  for (std::uint32_t key_lane = 0; key_lane < qw38::cuda::kGdnStateHeadWidth;
       ++key_lane) {
    const std::size_t index =
        head_base + static_cast<std::size_t>(key_lane) *
                        qw38::cuda::kGdnStateHeadWidth +
        lane;
    const float current = qw38::cuda::gdn_state_bf16_to_fp32(source[index]);
    prediction = __fadd_rn(prediction, __fmul_rn(current, decay));
  }
  const float delta = __fmul_rn(value[lane] - prediction, beta);
  float result = 0.0F;
  for (std::uint32_t key_lane = 0; key_lane < qw38::cuda::kGdnStateHeadWidth;
       ++key_lane) {
    const std::size_t index =
        head_base + static_cast<std::size_t>(key_lane) *
                        qw38::cuda::kGdnStateHeadWidth +
        lane;
    const float current = qw38::cuda::gdn_state_bf16_to_fp32(source[index]);
    const float updated =
        __fadd_rn(__fmul_rn(current, decay), __fmul_rn(key[key_lane], delta));
    candidate[index] = qw38::cuda::gdn_state_fp32_to_bf16(updated);
    result = __fadd_rn(result, __fmul_rn(query[key_lane], updated));
  }
  output[static_cast<std::size_t>(value_head) * qw38::cuda::kGdnStateHeadWidth +
         lane] = result;
}

float event_ms(cudaEvent_t start, cudaEvent_t stop) {
  float ms = 0.0F;
  cudaEventElapsedTime(&ms, start, stop);
  return ms;
}

int run_reference() {
  constexpr std::size_t kCount = 256;
  std::vector<float> source(kCount);
  std::vector<float> restored_bf16(kCount);
  std::vector<float> restored_q8(kCount);
  std::vector<__nv_bfloat16> packed_bf16(kCount);
  std::vector<std::int8_t> packed_q8(kCount);
  std::vector<__half> scales(qw38::cuda::gdn_state_q8_groups(kCount));
  for (std::size_t index = 0; index < kCount; ++index) {
    const float unit = static_cast<float>(index) / 32.0F;
    source[index] = (index % 17 == 0) ? 0.0F : std::sin(unit) * 8.0F;
  }
  source[1] = 1.0e-8F;
  source[2] = 0.999F;
  source[3] = 12.5F;
  source[4] = -7.25F;
  source[kCount - 1] = 1.0e6F;
  qw38::cuda::gdn_state_pack_bf16_host(source.data(), packed_bf16.data(),
                                       kCount);
  qw38::cuda::gdn_state_unpack_bf16_host(packed_bf16.data(),
                                         restored_bf16.data(), kCount);
  qw38::cuda::gdn_state_pack_q8_host(source.data(), packed_q8.data(),
                                     scales.data(), kCount);
  qw38::cuda::gdn_state_unpack_q8_host(packed_q8.data(), scales.data(),
                                       restored_q8.data(), kCount);
  double bf16_sq = 0.0;
  double q8_sq = 0.0;
  constexpr std::size_t kRmsCount = 224;
  bool finite = true;
  for (std::size_t index = 0; index < kCount; ++index) {
    finite = finite && std::isfinite(restored_bf16[index]) &&
             std::isfinite(restored_q8[index]);
    if (index >= kRmsCount) continue;
    const double err_b =
        static_cast<double>(restored_bf16[index] - source[index]);
    const double err_q =
        static_cast<double>(restored_q8[index] - source[index]);
    bf16_sq += err_b * err_b;
    q8_sq += err_q * err_q;
  }
  const double bf16_rms = std::sqrt(bf16_sq / static_cast<double>(kRmsCount));
  const double q8_rms = std::sqrt(q8_sq / static_cast<double>(kRmsCount));
  const bool zeros_ok = restored_bf16[0] == 0.0F && packed_q8[0] == 0;
  const bool near_unit_ok = std::fabs(restored_bf16[2] - source[2]) < 0.01F;
  const bool small_ok = std::fabs(restored_bf16[1]) < 1.0e-6F;
  const bool outlier_finite = std::isfinite(restored_bf16[kCount - 1]) &&
                              std::isfinite(restored_q8[kCount - 1]);
  const bool ok = finite && zeros_ok && near_unit_ok && small_ok &&
                  outlier_finite && bf16_rms < 0.05 && q8_rms < 0.25;
  const std::size_t fp32_bytes =
      qw38::cuda::gdn_state_session_bytes(qw38::cuda::GdnStateFormat::kFp32);
  const std::size_t bf16_bytes =
      qw38::cuda::gdn_state_session_bytes(qw38::cuda::GdnStateFormat::kBf16);
  const std::size_t q8_bytes =
      qw38::cuda::gdn_state_session_bytes(qw38::cuda::GdnStateFormat::kQ8Block32);
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-122\",\"workload\":\"reference\","
      "\"ok\":%s,\"finite\":%s,\"zeros_ok\":%s,\"near_unit_ok\":%s,"
      "\"small_update_ok\":%s,\"outlier_finite\":%s,"
      "\"bf16_rms\":%.9g,\"q8_rms\":%.9g,\"q8_group\":%d,"
      "\"fp32_session_bytes\":%zu,\"bf16_session_bytes\":%zu,"
      "\"q8_session_bytes\":%zu,\"convolution_fp32\":true,"
      "\"no_fp32_shadow\":true,\"keep\":false}\n",
      kPrefix, json_bool(ok), json_bool(finite), json_bool(zeros_ok),
      json_bool(near_unit_ok), json_bool(small_ok), json_bool(outlier_finite),
      bf16_rms, q8_rms, qw38::cuda::kGdnStateQ8Group, fp32_bytes, bf16_bytes,
      q8_bytes);
  print_counts(0, 1, 2, false);
  return ok ? 0 : 1;
}

float time_layer_launches(bool bf16, float* fp32_src, float* fp32_dst,
                          __nv_bfloat16* bf_src, __nv_bfloat16* bf_dst,
                          float* output, float* query, float* key, float* value,
                          cudaStream_t stream) {
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  for (int warm = 0; warm < kRecurrenceWarmups; ++warm) {
    for (unsigned layer = 0; layer < qw38::cuda::kGdnStateLayers; ++layer) {
      const std::size_t offset =
          static_cast<std::size_t>(layer) * qw38::cuda::kGdnStateRecurrentValues;
      if (bf16) {
        opt122_recurrence_bf16<<<qw38::cuda::kGdnStateValueHeads,
                                 qw38::cuda::kGdnStateHeadWidth, 0, stream>>>(
            bf_src + offset, bf_dst + offset, output, query, key, value, 0.97F,
            0.15F);
      } else {
        opt122_recurrence_fp32<<<qw38::cuda::kGdnStateValueHeads,
                                 qw38::cuda::kGdnStateHeadWidth, 0, stream>>>(
            fp32_src + offset, fp32_dst + offset, output, query, key, value,
            0.97F, 0.15F);
      }
    }
  }
  cudaStreamSynchronize(stream);
  cudaEventRecord(start, stream);
  for (int iter = 0; iter < kRecurrenceIters; ++iter) {
    for (unsigned layer = 0; layer < qw38::cuda::kGdnStateLayers; ++layer) {
      const std::size_t offset =
          static_cast<std::size_t>(layer) * qw38::cuda::kGdnStateRecurrentValues;
      if (bf16) {
        opt122_recurrence_bf16<<<qw38::cuda::kGdnStateValueHeads,
                                 qw38::cuda::kGdnStateHeadWidth, 0, stream>>>(
            bf_src + offset, bf_dst + offset, output, query, key, value, 0.97F,
            0.15F);
      } else {
        opt122_recurrence_fp32<<<qw38::cuda::kGdnStateValueHeads,
                                 qw38::cuda::kGdnStateHeadWidth, 0, stream>>>(
            fp32_src + offset, fp32_dst + offset, output, query, key, value,
            0.97F, 0.15F);
      }
    }
  }
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  const float total = event_ms(start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return total / static_cast<float>(kRecurrenceIters);
}

int run_inventory(const qw38::cuda::ResidentModel& model,
                  const Options& options) {
  const std::size_t prefix = options.prefix == 0 ? 128 : options.prefix;
  const std::size_t capacity = session_capacity(prefix, 2, options.capacity);
  std::vector<std::size_t> tokens(prefix + 2);
  fill_tokens(&tokens);
  qw38::cuda::ExecutionGraphPathScope scope(options.execution_graphs);
  qw38::cuda::SchedulerSession session;
  qw38::cuda::SchedulerWorkspace workspace;
  qw38::cuda::SchedulerGraphs graphs;
  qw38::Status status = session.create(capacity);
  if (status.is_ok()) status = workspace.create(capacity);
  if (status.is_ok()) status = create_graphs(model, &workspace, &graphs);
  std::vector<float> logits(qw38::internal::kVocabularySize);
  std::array<float, qw38::internal::kResidualWidth> hidden{};
  qw38::cuda::SyncResult sync{};
  if (status.is_ok()) {
    status = qw38::cuda::sync_tokens(
        model, tokens.data(), prefix, &session, &workspace, logits.data(),
        logits.size(), hidden.data(), hidden.size(), &sync, nullptr, &graphs);
  }
  workspace.reset_transfer_counters();
  qw38::cuda::DecodeAttribution attribution{};
  float elapsed = 0.0F;
  if (status.is_ok()) {
    status = qw38::cuda::execute_token(
        model, tokens[prefix], &session, &workspace, logits.data(),
        logits.size(), hidden.data(), hidden.size(), &elapsed, nullptr, nullptr,
        qw38::cuda::PointwisePath::kFused, &graphs, &attribution);
  }
  if (!status.is_ok()) return fail_status(status);
  const std::uint64_t decode_d2d = workspace.transfer_d2d_bytes_;
  const std::uint64_t decode_d2h = workspace.transfer_d2h_bytes_;
  const float gdn_core_ms = attribution.gdn_core.milliseconds;
  const bool gdn_measured = attribution.gdn_core.measured;
  const float wall_ms = attribution.wall.milliseconds;
  const float mixer_ms = attribution.mixer_mmv.milliseconds;
  const float ffn_ms = attribution.ffn_mmv.milliseconds;

  const std::size_t rec_fp32 =
      qw38::cuda::gdn_state_recurrent_payload_bytes(
          qw38::cuda::GdnStateFormat::kFp32);
  const std::size_t rec_bf16 =
      qw38::cuda::gdn_state_recurrent_payload_bytes(
          qw38::cuda::GdnStateFormat::kBf16);
  const std::size_t rec_q8 = qw38::cuda::gdn_state_recurrent_payload_bytes(
      qw38::cuda::GdnStateFormat::kQ8Block32);
  const std::size_t conv = qw38::cuda::gdn_state_convolution_payload_bytes();
  const std::size_t dram_fp32 = qw38::cuda::gdn_state_decode_dram_bytes(
      qw38::cuda::GdnStateFormat::kFp32);
  const std::size_t dram_bf16 = qw38::cuda::gdn_state_decode_dram_bytes(
      qw38::cuda::GdnStateFormat::kBf16);
  const std::size_t dram_q8 = qw38::cuda::gdn_state_decode_dram_bytes(
      qw38::cuda::GdnStateFormat::kQ8Block32);
  const std::size_t amp_fp32 = qw38::cuda::gdn_state_decode_register_bytes(
      qw38::cuda::GdnStateFormat::kFp32);
  const std::size_t amp_bf16 = qw38::cuda::gdn_state_decode_register_bytes(
      qw38::cuda::GdnStateFormat::kBf16);

  float* fp32_src = nullptr;
  float* fp32_dst = nullptr;
  __nv_bfloat16* bf_src = nullptr;
  __nv_bfloat16* bf_dst = nullptr;
  float* output = nullptr;
  float* query = nullptr;
  float* key = nullptr;
  float* value = nullptr;
  cudaMalloc(&fp32_src, rec_fp32);
  cudaMalloc(&fp32_dst, rec_fp32);
  cudaMalloc(&bf_src, rec_bf16);
  cudaMalloc(&bf_dst, rec_bf16);
  cudaMalloc(&output, qw38::internal::kGdnValueWidth * sizeof(float));
  cudaMalloc(&query, qw38::cuda::kGdnStateHeadWidth * sizeof(float));
  cudaMalloc(&key, qw38::cuda::kGdnStateHeadWidth * sizeof(float));
  cudaMalloc(&value, qw38::cuda::kGdnStateHeadWidth * sizeof(float));
  cudaMemset(fp32_src, 0, rec_fp32);
  cudaMemset(bf_src, 0, rec_bf16);
  cudaMemset(query, 0, qw38::cuda::kGdnStateHeadWidth * sizeof(float));
  cudaMemset(key, 0, qw38::cuda::kGdnStateHeadWidth * sizeof(float));
  cudaMemset(value, 0, qw38::cuda::kGdnStateHeadWidth * sizeof(float));

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaDeviceSynchronize();
  cudaEventRecord(start);
  cudaMemcpy(fp32_dst, fp32_src, rec_fp32, cudaMemcpyDeviceToDevice);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  const float memcpy_fp32_ms = event_ms(start, stop);
  cudaEventRecord(start);
  cudaMemcpy(bf_dst, bf_src, rec_bf16, cudaMemcpyDeviceToDevice);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  const float memcpy_bf16_ms = event_ms(start, stop);
  cudaEventRecord(start);
  cudaMemcpy(fp32_dst, fp32_src, rec_fp32, cudaMemcpyDeviceToDevice);
  cudaMemcpy(fp32_dst, fp32_src, std::min(rec_fp32, conv),
             cudaMemcpyDeviceToDevice);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  const float memcpy_state_ms = event_ms(start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);
  const float fp32_rec_ms =
      time_layer_launches(false, fp32_src, fp32_dst, bf_src, bf_dst, output,
                          query, key, value, stream);
  const float bf16_rec_ms =
      time_layer_launches(true, fp32_src, fp32_dst, bf_src, bf_dst, output,
                          query, key, value, stream);
  cudaStreamDestroy(stream);
  cudaFree(fp32_src);
  cudaFree(fp32_dst);
  cudaFree(bf_src);
  cudaFree(bf_dst);
  cudaFree(output);
  cudaFree(query);
  cudaFree(key);
  cudaFree(value);

  const float peak_fp32_ms = qw38::cuda::gdn_state_bytes_to_peak_ms(dram_fp32);
  const float peak_bf16_ms = qw38::cuda::gdn_state_bytes_to_peak_ms(dram_bf16);
  const float peak_q8_ms = qw38::cuda::gdn_state_bytes_to_peak_ms(dram_q8);
  const bool carry_production = false;
  const std::size_t prompt_chunk = qw38::cuda::kPromptChunkRows;
  const std::size_t prompt_micro = qw38::cuda::selected_prompt_microbatch_rows();
  const bool ok = gdn_measured && dram_fp32 > 0;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-122\",\"workload\":\"inventory\","
      "\"ok\":%s,\"parent\":\"post113_selected\","
      "\"execution_graphs\":\"ffn_only\",\"prefix\":%zu,"
      "\"gdn_core_ms\":%.9g,\"gdn_core_measured\":%s,\"wall_ms\":%.9g,"
      "\"mixer_mmv_ms\":%.9g,\"ffn_mmv_ms\":%.9g,\"token_elapsed_ms\":%.9g,"
      "\"decode_d2d_bytes\":%llu,\"decode_d2h_bytes\":%llu,"
      "\"gdn_carry_production_traffic\":%s,\"prompt_chunk_rows\":%zu,"
      "\"prompt_microbatch_rows\":%zu,\"pointer_swap_commit\":true,"
      "\"recurrent_fp32_bytes\":%zu,\"recurrent_bf16_bytes\":%zu,"
      "\"recurrent_q8_bytes\":%zu,\"convolution_fp32_bytes\":%zu,"
      "\"dram_fp32_bytes\":%zu,\"dram_bf16_bytes\":%zu,\"dram_q8_bytes\":%zu,"
      "\"register_amp_fp32_bytes\":%zu,\"register_amp_bf16_bytes\":%zu,"
      "\"l2_covers_second_read\":true,\"peak_fp32_ms\":%.9g,"
      "\"peak_bf16_ms\":%.9g,\"peak_q8_ms\":%.9g,"
      "\"memcpy_rec_fp32_ms\":%.9g,\"memcpy_rec_bf16_ms\":%.9g,"
      "\"memcpy_state_fp32_ms\":%.9g,\"fp32_recurrence_ms\":%.9g,"
      "\"bf16_recurrence_ms\":%.9g,\"opt110_hooks_ready\":%s,\"keep\":false}\n",
      kPrefix, json_bool(ok), prefix, static_cast<double>(gdn_core_ms),
      json_bool(gdn_measured), static_cast<double>(wall_ms),
      static_cast<double>(mixer_ms), static_cast<double>(ffn_ms),
      static_cast<double>(elapsed),
      static_cast<unsigned long long>(decode_d2d),
      static_cast<unsigned long long>(decode_d2h), json_bool(carry_production),
      prompt_chunk, prompt_micro, rec_fp32, rec_bf16, rec_q8, conv, dram_fp32,
      dram_bf16, dram_q8, amp_fp32, amp_bf16, static_cast<double>(peak_fp32_ms),
      static_cast<double>(peak_bf16_ms), static_cast<double>(peak_q8_ms),
      static_cast<double>(memcpy_fp32_ms), static_cast<double>(memcpy_bf16_ms),
      static_cast<double>(memcpy_state_ms), static_cast<double>(fp32_rec_ms),
      static_cast<double>(bf16_rec_ms),
      json_bool(qw38::cuda::opt110::engine_hooks_ready()));
  print_counts(1, 1, 2, false);
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (!qw38::cuda::test_tier_valid()) {
    std::fprintf(stderr, "QW38_CUDA_TEST_TIER must be set\n");
    return 1;
  }
  Options options;
  const int model_index = parse_args(argc, argv, &options);
  if (model_index < 0) return model_index == 0 ? 1 : model_index;

  if (std::strcmp(options.workload, "reference") == 0) {
    return run_reference();
  }

  qw38::internal::MappedFile mapping;
  qw38::cuda::ResidentModel model;
  const qw38::Status loaded = load_model(argv[model_index], &mapping, &model);
  if (!loaded.is_ok()) return fail_status(loaded);

  cudaDeviceProp prop{};
  int device = 0;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  std::printf("device=%s compute=%d.%d opt110_engine_hooks_ready=%s\n",
              prop.name, prop.major, prop.minor,
              json_bool(qw38::cuda::opt110::engine_hooks_ready()));

  if (std::strcmp(options.workload, "inventory") == 0) {
    return run_inventory(model, options);
  }
  std::fprintf(stderr, "unknown workload %s\n", options.workload);
  return 2;
}
