#include "attention_decode.h"
#include "full_scheduler.h"
#include "gdn_step.h"
#include "mixer.h"
#include "model.h"
#include "pdl_launch.cuh"
#include "quant_mmv.h"
#include "scheduler.h"
#include "scheduler_primitives.h"
#include "weights.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

constexpr std::size_t kCapacity = 131072;
constexpr std::size_t kTokens = 4096;
constexpr int kReplicates = 3;
constexpr qw38::cuda::GdnConfig kGdnConfig{16, 48, 128, 128, 4};
const char* kIds[] = {"off", "pdl_host", "pdl"};

__global__ void pdl_ptx_probe_kernel() {}

const char* json_bool(bool value) { return value ? "true" : "false"; }

int fail_status(const qw38::Status& status) {
  std::fprintf(stderr, "%s\n", status.message().c_str());
  return 1;
}

int fail_cuda(const char* where, cudaError_t error) {
  std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(error));
  return 1;
}

bool bytes_equal(const void* left, const void* right, std::size_t bytes) {
  return std::memcmp(left, right, bytes) == 0;
}

int occupancy_ok() {
  if (qw38::cuda::gdn_fused_quality_occupancy() < 1) return 0;
  if (qw38::cuda::q8_quality_mmq_occupancy(128) < 1) return 0;
  if (qw38::cuda::attention_mma_quality_occupancy_for_path("stream_k") < 1) {
    return 0;
  }
  return 1;
}

const qw38::cuda::DeviceLayer* first_layer(const qw38::cuda::ResidentModel& model,
                                           qw38::internal::LayerKind kind) {
  for (std::size_t index = 0; index < model.layer_count(); ++index) {
    if (model.layer(index).kind == kind) return &model.layer(index);
  }
  return nullptr;
}

cudaError_t run_gdn_chain(const qw38::cuda::DeviceLayer& layer,
                          qw38::cuda::SchedulerWorkspace* workspace,
                          const qw38::cuda::GdnState& committed,
                          const qw38::cuda::GdnState& candidate,
                          std::size_t token_count, cudaStream_t stream) {
  cudaError_t error = qw38::cuda::launch_rms_norm_rows_fp32_to_bf16(
      workspace->prompt_residual_a_, layer.common.input_norm,
      qw38::internal::kResidualWidth, token_count,
      workspace->prompt_normalized_, stream);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_mmq_q8_1(
        qw38::cuda::QuantKind::kQ8_0, workspace->prompt_normalized_, token_count,
        qw38::internal::kResidualWidth, workspace->prompt_q8_, stream);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_q8_mmq_quality_mma(
        layer.gdn.packed_qkv.data, layer.gdn.packed_qkv.rows,
        layer.gdn.packed_qkv.columns, workspace->prompt_q8_, token_count,
        workspace->prompt_projection_a_, stream);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_q8_mmq_quality_mma(
        layer.gdn.value_gate.data, layer.gdn.value_gate.rows,
        layer.gdn.value_gate.columns, workspace->prompt_q8_, token_count,
        workspace->prompt_projection_b_, stream);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_q8_mmq_quality_mma(
        layer.gdn.alpha.data, layer.gdn.alpha.rows, layer.gdn.alpha.columns,
        workspace->prompt_q8_, token_count, workspace->prompt_projection_c_,
        stream);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_q8_mmq_quality_mma(
        layer.gdn.beta.data, layer.gdn.beta.rows, layer.gdn.beta.columns,
        workspace->prompt_q8_, token_count, workspace->prompt_projection_d_,
        stream);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_prepare_gdn_gate_rows(
        workspace->prompt_projection_c_, workspace->prompt_projection_d_,
        layer.gdn.folded_a, layer.gdn.dt_bias, 16, 3, token_count,
        workspace->prompt_gdn_decay_, workspace->prompt_gdn_update_, stream);
  }
  const std::size_t gdn_scratch_floats =
      workspace->prompt_chunk_rows_ * qw38::internal::kFfnWidth *
      sizeof(__nv_bfloat16) / sizeof(float);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_gdn_prepare_chunk_tiled(
        kGdnConfig, workspace->prompt_projection_a_, layer.gdn.convolution,
        workspace->prompt_gdn_decay_, workspace->prompt_gdn_update_, token_count,
        committed, candidate, workspace->prompt_gdn_convolved_,
        workspace->prompt_gdn_recurrent_output_, stream,
        qw38::cuda::GdnScanPath::kFusedTokenLoop,
        reinterpret_cast<float*>(workspace->prompt_projected_bf16_),
        gdn_scratch_floats);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_gdn_gated_output_rows(
        workspace->prompt_gdn_recurrent_output_, workspace->prompt_projection_b_,
        layer.gdn.norm, 16, 3, 128, token_count,
        workspace->prompt_projected_bf16_, stream);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_mmq_q8_1(
        qw38::cuda::QuantKind::kQ8_0, workspace->prompt_projected_bf16_,
        token_count, qw38::internal::kGdnValueWidth, workspace->prompt_q8_,
        stream);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_q8_mmq_quality_mma(
        layer.gdn.output.data, layer.gdn.output.rows, layer.gdn.output.columns,
        workspace->prompt_q8_, token_count, workspace->prompt_mixer_output_,
        stream);
  }
  return error;
}

cudaError_t run_attention_chain(const qw38::cuda::DeviceLayer& layer,
                                qw38::cuda::SchedulerWorkspace* workspace,
                                const qw38::cuda::AttentionCache& committed,
                                const qw38::cuda::AttentionCache& candidate,
                                std::size_t token_count, cudaStream_t stream) {
  cudaError_t error = qw38::cuda::launch_rms_norm_rows_fp32_to_bf16(
      workspace->prompt_residual_a_, layer.common.input_norm,
      qw38::internal::kResidualWidth, token_count,
      workspace->prompt_normalized_, stream);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quantize_mmq_q8_1(
        qw38::cuda::QuantKind::kQ8_0, workspace->prompt_normalized_, token_count,
        qw38::internal::kResidualWidth, workspace->prompt_q8_, stream);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_q8_mmq_quality_mma(
        layer.attention.query_gate.data, layer.attention.query_gate.rows,
        layer.attention.query_gate.columns, workspace->prompt_q8_, token_count,
        workspace->prompt_projection_a_, stream);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_q8_mmq_quality_mma(
        layer.attention.key.data, layer.attention.key.rows,
        layer.attention.key.columns, workspace->prompt_q8_, token_count,
        workspace->prompt_projection_c_, stream);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_q8_mmq_quality_mma(
        layer.attention.value.data, layer.attention.value.rows,
        layer.attention.value.columns, workspace->prompt_q8_, token_count,
        workspace->prompt_projection_d_, stream);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_split_attention_rows_prompt(
        workspace->prompt_projection_a_, qw38::internal::kAttentionQueryWidth,
        256, token_count, workspace->prompt_gdn_convolved_,
        workspace->prompt_projection_b_, stream);
  }
  const qw38::cuda::AttentionConfig config{24, 4, 256, 64, kCapacity};
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_attention_prepare_chunk_stream_k(
        config, 0, token_count, workspace->prompt_gdn_convolved_,
        workspace->prompt_projection_c_, workspace->prompt_projection_d_,
        layer.attention.query_norm, layer.attention.key_norm,
        workspace->prompt_projection_b_, committed, candidate,
        workspace->attention_normalized_query_,
        workspace->attention_normalized_key_, workspace->attention_scores_,
        workspace->prompt_gdn_recurrent_output_,
        reinterpret_cast<float*>(workspace->prompt_projected_bf16_),
        reinterpret_cast<float*>(workspace->prompt_q8_), stream);
  }
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_fp32_to_bf16(
        workspace->prompt_gdn_recurrent_output_,
        token_count * qw38::internal::kAttentionQueryWidth,
        workspace->prompt_projected_bf16_, stream);
  }
  if (error == cudaSuccess) {
    const std::size_t fixup_floats =
        workspace->prompt_chunk_rows_ * qw38::internal::kFfnWidth;
    error = qw38::cuda::launch_quant_mmq(
        layer.attention.output.kind, layer.attention.output.data,
        layer.attention.output.rows, layer.attention.output.columns,
        workspace->prompt_projected_bf16_, token_count, workspace->prompt_q8_,
        workspace->prompt_mixer_output_, stream, workspace->prompt_projection_a_,
        fixup_floats);
  }
  return error;
}

struct Snapshot {
  std::vector<float> gdn_mixer;
  std::vector<float> attn_mixer;
  std::vector<float> gdn_conv;
  std::vector<float> gdn_rec;
  std::vector<__nv_bfloat16> attn_key;
  std::vector<__nv_bfloat16> attn_value;
};

cudaError_t copy_floats(std::vector<float>* dest, const float* src,
                        std::size_t count) {
  dest->resize(count);
  return cudaMemcpy(dest->data(), src, count * sizeof(float),
                    cudaMemcpyDeviceToHost);
}

cudaError_t copy_bf16(std::vector<__nv_bfloat16>* dest,
                      const __nv_bfloat16* src, std::size_t count) {
  dest->resize(count);
  return cudaMemcpy(dest->data(), src, count * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToHost);
}

bool snapshot_equal(const Snapshot& left, const Snapshot& right) {
  return bytes_equal(left.gdn_mixer.data(), right.gdn_mixer.data(),
                     left.gdn_mixer.size() * sizeof(float)) &&
         bytes_equal(left.attn_mixer.data(), right.attn_mixer.data(),
                     left.attn_mixer.size() * sizeof(float)) &&
         bytes_equal(left.gdn_conv.data(), right.gdn_conv.data(),
                     left.gdn_conv.size() * sizeof(float)) &&
         bytes_equal(left.gdn_rec.data(), right.gdn_rec.data(),
                     left.gdn_rec.size() * sizeof(float)) &&
         bytes_equal(left.attn_key.data(), right.attn_key.data(),
                     left.attn_key.size() * sizeof(__nv_bfloat16)) &&
         bytes_equal(left.attn_value.data(), right.attn_value.data(),
                     left.attn_value.size() * sizeof(__nv_bfloat16));
}

bool snapshot_finite(const Snapshot& snap) {
  for (float value : snap.gdn_mixer) {
    if (!std::isfinite(value)) return false;
  }
  for (float value : snap.attn_mixer) {
    if (!std::isfinite(value)) return false;
  }
  for (float value : snap.gdn_conv) {
    if (!std::isfinite(value)) return false;
  }
  for (float value : snap.gdn_rec) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

void ensure_evidence_dir() {
  mkdir("evidence", 0755);
  mkdir("evidence/optimization", 0755);
  mkdir("evidence/optimization/opt030-pdl-launches", 0755);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
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
  if (!status.is_ok()) return fail_status(status);

  const qw38::cuda::DeviceLayer* gdn_layer =
      first_layer(model, qw38::internal::LayerKind::kGdn);
  const qw38::cuda::DeviceLayer* attn_layer =
      first_layer(model, qw38::internal::LayerKind::kAttention);
  if (gdn_layer == nullptr || attn_layer == nullptr) {
    std::fprintf(stderr, "model is missing GDN or attention layers\n");
    return 1;
  }

  std::vector<std::size_t> tokens(kTokens);
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    tokens[index] = (42 + index * 997) % qw38::internal::kVocabularySize;
  }

  qw38::cuda::SchedulerWorkspace workspace;
  status = workspace.create(kCapacity);
  if (!status.is_ok()) return fail_status(status);

  cudaStream_t stream = nullptr;
  cudaError_t error =
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (error != cudaSuccess) return fail_cuda("stream", error);

  error = cudaMemcpyAsync(
      reinterpret_cast<std::size_t*>(workspace.prompt_q8_), tokens.data(),
      kTokens * sizeof(std::size_t), cudaMemcpyHostToDevice, stream);
  if (error == cudaSuccess) {
    error = qw38::cuda::launch_quant_rows_decode_widen(
        model.embedding().kind, model.embedding().data, model.embedding().rows,
        model.embedding().columns,
        reinterpret_cast<const std::size_t*>(workspace.prompt_q8_), kTokens,
        workspace.prompt_residual_a_, stream);
  }
  if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
  if (error != cudaSuccess) return fail_cuda("embedding", error);

  std::vector<float> residual_host(kTokens * qw38::internal::kResidualWidth);
  error = cudaMemcpy(residual_host.data(), workspace.prompt_residual_a_,
                     residual_host.size() * sizeof(float),
                     cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return fail_cuda("residual snapshot", error);

  float* gdn_committed_conv = nullptr;
  float* gdn_committed_rec = nullptr;
  float* gdn_candidate_conv = nullptr;
  float* gdn_candidate_rec = nullptr;
  __nv_bfloat16* attn_committed_key = nullptr;
  __nv_bfloat16* attn_committed_value = nullptr;
  __nv_bfloat16* attn_candidate_key = nullptr;
  __nv_bfloat16* attn_candidate_value = nullptr;
  const std::size_t conv_bytes =
      qw38::internal::kGdnConvolutionValues * sizeof(float);
  const std::size_t rec_bytes =
      qw38::internal::kGdnRecurrentStateValues * sizeof(float);
  const std::size_t kv_bytes =
      kTokens * qw38::internal::kAttentionKvWidth * sizeof(__nv_bfloat16);
  error = cudaMalloc(&gdn_committed_conv, conv_bytes);
  if (error == cudaSuccess) error = cudaMalloc(&gdn_committed_rec, rec_bytes);
  if (error == cudaSuccess) error = cudaMalloc(&gdn_candidate_conv, conv_bytes);
  if (error == cudaSuccess) error = cudaMalloc(&gdn_candidate_rec, rec_bytes);
  if (error == cudaSuccess) error = cudaMalloc(&attn_committed_key, kv_bytes);
  if (error == cudaSuccess) {
    error = cudaMalloc(&attn_committed_value, kv_bytes);
  }
  if (error == cudaSuccess) error = cudaMalloc(&attn_candidate_key, kv_bytes);
  if (error == cudaSuccess) error = cudaMalloc(&attn_candidate_value, kv_bytes);
  if (error != cudaSuccess) return fail_cuda("A/B scratch", error);
  const qw38::cuda::GdnState gdn_committed{gdn_committed_conv, gdn_committed_rec};
  const qw38::cuda::GdnState gdn_candidate{gdn_candidate_conv, gdn_candidate_rec};
  const qw38::cuda::AttentionCache attn_committed{attn_committed_key,
                                                  attn_committed_value};
  const qw38::cuda::AttentionCache attn_candidate{attn_candidate_key,
                                                  attn_candidate_value};

  ensure_evidence_dir();
  FILE* raw = std::fopen(
      "evidence/optimization/opt030-pdl-launches/pdl-ab-raw.txt", "w");
  if (raw == nullptr) {
    std::fprintf(stderr, "cannot open pdl-ab-raw.txt\n");
    return 1;
  }

  const int occ = occupancy_ok();
  const bool ptx_ok = qw38::cuda::pdl_kernel_can_use(
      reinterpret_cast<const void*>(pdl_ptx_probe_kernel));
  std::fprintf(raw, "pdl_occupancy ok=%s gdn=%d q8=%d fattn=%d ptx=%s\n",
               json_bool(occ >= 1), qw38::cuda::gdn_fused_quality_occupancy(),
               qw38::cuda::q8_quality_mmq_occupancy(128),
               qw38::cuda::attention_mma_quality_occupancy_for_path("stream_k"),
               json_bool(ptx_ok));
  std::printf("pdl_occupancy ok=%s ptx=%s\n", json_bool(occ >= 1),
              json_bool(ptx_ok));

  cudaStream_t capture_stream = nullptr;
  error = cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking);
  bool capture_ok = false;
  bool capture_used_ex = true;
  if (error == cudaSuccess) {
    qw38::cuda::quartz_set_pdl_path_override("pdl_host");
    error = cudaStreamBeginCapture(capture_stream,
                                   cudaStreamCaptureModeThreadLocal);
    if (error == cudaSuccess) {
      const qw38::cuda::PdlScope scope;
      error = qw38::cuda::launch_rms_norm_rows_fp32_to_bf16(
          workspace.prompt_residual_a_, gdn_layer->common.input_norm,
          qw38::internal::kResidualWidth, 8, workspace.prompt_normalized_,
          capture_stream);
      capture_used_ex = qw38::cuda::pdl_last_used_launch_ex();
    }
    cudaGraph_t graph = nullptr;
    if (error == cudaSuccess) {
      error = cudaStreamEndCapture(capture_stream, &graph);
    }
    capture_ok = error == cudaSuccess && !capture_used_ex;
    if (graph != nullptr) cudaGraphDestroy(graph);
    qw38::cuda::quartz_set_pdl_path_override(nullptr);
  }
  if (capture_stream != nullptr) cudaStreamDestroy(capture_stream);
  std::fprintf(raw, "pdl_capture ordinary=%s used_ex=%s\n", json_bool(capture_ok),
               json_bool(capture_used_ex));
  std::printf("pdl_capture ordinary=%s used_ex=%s\n", json_bool(capture_ok),
              json_bool(capture_used_ex));

  Snapshot baseline;
  float mean_ms[3] = {0.0F, 0.0F, 0.0F};
  bool eligible[3] = {false, false, false};
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error != cudaSuccess) return fail_cuda("events", error);

  const std::size_t mixer_n = kTokens * qw38::internal::kResidualWidth;
  const std::size_t conv_n = qw38::internal::kGdnConvolutionValues;
  const std::size_t rec_n = qw38::internal::kGdnRecurrentStateValues;
  const std::size_t kv_n = kTokens * qw38::internal::kAttentionKvWidth;

  bool ab_ok = occ >= 1 && capture_ok && ptx_ok;
  for (int id = 0; id < 3 && ab_ok; ++id) {
    qw38::cuda::quartz_set_pdl_path_override(kIds[id]);
    error = qw38::cuda::quartz_set_pdl_device_ops(std::strcmp(kIds[id], "pdl") ==
                                                 0);
    eligible[id] = error == cudaSuccess && occ >= 1 && ptx_ok;
    float samples[3] = {0.0F, 0.0F, 0.0F};
    Snapshot snap;
    for (int sample = 0; sample < kReplicates && error == cudaSuccess; ++sample) {
      error = cudaMemcpy(workspace.prompt_residual_a_, residual_host.data(),
                         residual_host.size() * sizeof(float),
                         cudaMemcpyHostToDevice);
      if (error == cudaSuccess) error = cudaMemset(gdn_committed_conv, 0, conv_bytes);
      if (error == cudaSuccess) error = cudaMemset(gdn_committed_rec, 0, rec_bytes);
      if (error == cudaSuccess) error = cudaMemset(attn_committed_key, 0, kv_bytes);
      if (error == cudaSuccess) {
        error = cudaMemset(attn_committed_value, 0, kv_bytes);
      }
      if (error == cudaSuccess) {
        const qw38::cuda::PdlScope scope;
        error = cudaEventRecord(start, stream);
        if (error == cudaSuccess) {
          error = run_gdn_chain(*gdn_layer, &workspace, gdn_committed,
                                gdn_candidate, kTokens, stream);
        }
        if (error == cudaSuccess) {
          error = run_attention_chain(*attn_layer, &workspace, attn_committed,
                                      attn_candidate, kTokens, stream);
        }
        if (error == cudaSuccess) error = cudaEventRecord(stop, stream);
        if (error == cudaSuccess) error = cudaEventSynchronize(stop);
        if (error == cudaSuccess) {
          error = cudaEventElapsedTime(&samples[sample], start, stop);
        }
      }
      std::fprintf(raw, "pdl_ab id=%s sample=%d ms=%.9g occupancy=%d\n",
                   kIds[id], sample, samples[sample], occ);
      std::printf("pdl_ab id=%s sample=%d ms=%.9g occupancy=%d\n", kIds[id],
                  sample, samples[sample], occ);
    }
    if (error == cudaSuccess) {
      mean_ms[id] = (samples[0] + samples[1] + samples[2]) / 3.0F;
      error = copy_floats(&snap.gdn_mixer, workspace.prompt_mixer_output_,
                          mixer_n);
      // GDN mixer was overwritten by attention; recapture GDN outputs from
      // candidate conv/recurrent which the GDN chain wrote before attention.
      if (error == cudaSuccess) {
        error = copy_floats(&snap.attn_mixer, workspace.prompt_mixer_output_,
                            mixer_n);
      }
      if (error == cudaSuccess) {
        error = copy_floats(&snap.gdn_conv, gdn_candidate.convolution, conv_n);
      }
      if (error == cudaSuccess) {
        error = copy_floats(&snap.gdn_rec, gdn_candidate.recurrent, rec_n);
      }
      if (error == cudaSuccess) {
        error = copy_bf16(&snap.attn_key, attn_candidate.key, kv_n);
      }
      if (error == cudaSuccess) {
        error = copy_bf16(&snap.attn_value, attn_candidate.value, kv_n);
      }
    }
    // Re-run GDN-only after the timed chain to snapshot GDN mixer without
    // changing the timed samples. Restore residual and committed GDN state.
    if (error == cudaSuccess) {
      error = cudaMemcpy(workspace.prompt_residual_a_, residual_host.data(),
                         residual_host.size() * sizeof(float),
                         cudaMemcpyHostToDevice);
      if (error == cudaSuccess) error = cudaMemset(gdn_committed_conv, 0, conv_bytes);
      if (error == cudaSuccess) error = cudaMemset(gdn_committed_rec, 0, rec_bytes);
      if (error == cudaSuccess) {
        const qw38::cuda::PdlScope scope;
        error = run_gdn_chain(*gdn_layer, &workspace, gdn_committed,
                              gdn_candidate, kTokens, stream);
      }
      if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
      if (error == cudaSuccess) {
        error = copy_floats(&snap.gdn_mixer, workspace.prompt_mixer_output_,
                            mixer_n);
      }
      if (error == cudaSuccess) {
        error = copy_floats(&snap.gdn_conv, gdn_candidate.convolution, conv_n);
      }
      if (error == cudaSuccess) {
        error = copy_floats(&snap.gdn_rec, gdn_candidate.recurrent, rec_n);
      }
    }
    bool equal = true;
    if (error == cudaSuccess && id == 0) {
      baseline = snap;
      eligible[id] = eligible[id] && snapshot_finite(snap);
    } else if (error == cudaSuccess) {
      equal = snapshot_equal(snap, baseline);
      eligible[id] = eligible[id] && snapshot_finite(snap) && equal;
    }
    std::fprintf(raw,
                 "pdl_ab_mean id=%s mean_ms=%.9g occupancy=%d eligible=%s "
                 "byte_equal=%s\n",
                 kIds[id], mean_ms[id], occ, json_bool(eligible[id]),
                 json_bool(id == 0 || equal));
    std::printf("pdl_ab_mean id=%s mean_ms=%.9g occupancy=%d eligible=%s\n",
                kIds[id], mean_ms[id], occ, json_bool(eligible[id]));
    if (error != cudaSuccess) {
      fail_cuda("pdl A/B", error);
      ab_ok = false;
    }
  }
  qw38::cuda::quartz_set_pdl_path_override(nullptr);
  qw38::cuda::quartz_set_pdl_device_ops(false);

  const char* winner = "off";
  bool win = false;
  float best = mean_ms[0];
  if (ab_ok && eligible[0] && mean_ms[0] > 0.0F) {
    for (int id = 1; id < 3; ++id) {
      if (!eligible[id] || mean_ms[id] <= 0.0F) continue;
      if (mean_ms[id] < mean_ms[0] && (!win || mean_ms[id] < best)) {
        best = mean_ms[id];
        winner = kIds[id];
        win = true;
      }
    }
  }
  std::fprintf(raw,
               "pdl_ab_winner id=%s mean_ms=%.9g baseline_id=off "
               "baseline_ms=%.9g win=%s selected_path=%s\n",
               winner, win ? best : mean_ms[0], mean_ms[0], json_bool(win),
               qw38::cuda::selected_pdl_path());
  std::fclose(raw);
  std::printf(
      "pdl_ab_winner id=%s mean_ms=%.9g baseline_id=off "
      "baseline_ms=%.9g win=%s selected_path=%s\n",
      winner, win ? best : mean_ms[0], mean_ms[0], json_bool(win),
      qw38::cuda::selected_pdl_path());

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  cudaFree(gdn_committed_conv);
  cudaFree(gdn_committed_rec);
  cudaFree(gdn_candidate_conv);
  cudaFree(gdn_candidate_rec);
  cudaFree(attn_committed_key);
  cudaFree(attn_committed_value);
  cudaFree(attn_candidate_key);
  cudaFree(attn_candidate_value);
  if (!ab_ok || !eligible[0]) {
    std::fprintf(stderr, "OPT-030 A/B baseline is not eligible\n");
    return 1;
  }
  std::printf("status=passed\n");
  return 0;
}
