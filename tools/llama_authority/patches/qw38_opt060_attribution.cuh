#pragma once

// Private OPT-060 instrumentation. Not an upstream llama.cpp interface.
// Enabled only for the isolated diagnostic worktree via -DQW38_OPT060_ATTRIBUTION.

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

struct ggml_tensor;
struct ggml_cgraph;
struct ggml_backend_cuda_context;

namespace qw38_opt060 {

constexpr std::size_t kPool = 8192;
constexpr std::size_t kName = 96;
// One eager decode token is bounded well below this; drain after each token
// instead of enlarging the pool and keeping warmup records.

struct Record {
  char engine[16] = "llama";
  char phase[16]{};
  int sequence_position = -1;
  int token_position = -1;
  int layer = -1;
  char role[64]{};
  char tensor_name[kName]{};
  char weight_name[kName]{};
  char ggml_op[32]{};
  char tensor_type[24]{};
  int m = 0;
  int n = 0;
  int k = 0;
  int64_t strides[4]{};
  unsigned long long stream = 0;
  int stream_index = 0;
  char launch_family[32]{};
  char fused_member_ids[192]{};
  int fused_member_count = 1;
  int start_event_id = -1;
  int end_event_id = -1;
  int scope_id = -1;
  int parent_scope_id = -1;
  char graph_mode[32]{};
  char attribution_role[16]{};
  float start_ms = 0.0F;
  float end_ms = 0.0F;
  float complete_work_ms = 0.0F;
  bool attributed = false;
  bool pool_overflow = false;
};

inline void copy_str(char* dest, std::size_t n, const char* text) {
  if (dest == nullptr || n == 0) return;
  if (text == nullptr) {
    dest[0] = '\0';
    return;
  }
  std::snprintf(dest, n, "%s", text);
}

inline const char* env_or(const char* name, const char* fallback) {
  const char* value = std::getenv(name);
  return (value != nullptr && value[0] != '\0') ? value : fallback;
}

inline bool env_flag(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

inline bool enabled() { return env_flag("QW38_OPT060_ATTRIBUTION"); }

inline bool force_eager() {
  return env_flag("QW38_OPT060_EAGER") || env_flag("QW38_OPT060_DIAGNOSTIC_EAGER");
}

inline const char* graph_mode_label() {
  if (force_eager()) return "eager_diagnostic";
  if (env_flag("GGML_CUDA_DISABLE_GRAPHS")) return "eager_diagnostic";
  return "cuda_graph";
}

struct Pool {
  std::array<cudaEvent_t, kPool> start{};
  std::array<cudaEvent_t, kPool> stop{};
  std::array<cudaStream_t, kPool> streams{};
  std::array<Record, kPool> meta{};
  std::vector<Record> archive{};
  cudaEvent_t epoch = nullptr;
  std::size_t created = 0;
  std::size_t count = 0;
  int next_scope_id = 1;
  int token_scope_id = 0;
  bool active = false;
  bool overflow = false;
  bool capturing = false;
  bool recording = false;
  char launch_family[32]{};
  int token_position = -1;
  int sequence_position = -1;
  int last_layer = -1;
  char phase[16]{};
  std::mutex mutex;

  ~Pool() { destroy(); }

  void destroy() {
    for (std::size_t i = 0; i < created; ++i) {
      if (stop[i] != nullptr) cudaEventDestroy(stop[i]);
      if (start[i] != nullptr) cudaEventDestroy(start[i]);
      start[i] = nullptr;
      stop[i] = nullptr;
    }
    if (epoch != nullptr) cudaEventDestroy(epoch);
    epoch = nullptr;
    created = 0;
    count = 0;
    archive.clear();
    active = false;
    recording = false;
    overflow = false;
  }

  cudaError_t ensure_epoch() {
    if (epoch != nullptr) return cudaSuccess;
    return cudaEventCreate(&epoch);
  }

  cudaError_t ensure_slot(std::size_t index) {
    if (index >= kPool) {
      overflow = true;
      return cudaErrorInvalidValue;
    }
    cudaError_t error = cudaSuccess;
    if (index >= created) {
      error = cudaEventCreate(&start[index]);
      if (error == cudaSuccess) error = cudaEventCreate(&stop[index]);
      if (error == cudaSuccess) created = index + 1;
    }
    return error;
  }
};

inline Pool& pool() {
  static Pool instance;
  return instance;
}

inline void set_launch_family(const char* family) {
  if (!enabled()) return;
  copy_str(pool().launch_family, sizeof(pool().launch_family), family);
}

inline void set_positions(const char* phase, int sequence, int token) {
  Pool& p = pool();
  std::lock_guard<std::mutex> lock(p.mutex);
  copy_str(p.phase, sizeof(p.phase), phase);
  p.sequence_position = sequence;
  p.token_position = token;
  p.last_layer = -1;
  p.token_scope_id = p.next_scope_id++;
}

inline void begin_run() {
  if (!enabled()) return;
  Pool& p = pool();
  std::lock_guard<std::mutex> lock(p.mutex);
  p.count = 0;
  p.archive.clear();
  p.active = false;
  p.overflow = false;
  p.capturing = false;
  p.recording = false;
  p.next_scope_id = 1;
  p.token_scope_id = 0;
  p.last_layer = -1;
  p.launch_family[0] = '\0';
}

inline void set_recording(bool enabled_flag) {
  Pool& p = pool();
  std::lock_guard<std::mutex> lock(p.mutex);
  p.recording = enabled_flag;
}

inline void begin_measured_window(cudaStream_t stream) {
  if (!enabled()) return;
  cudaDeviceSynchronize();
  Pool& p = pool();
  std::lock_guard<std::mutex> lock(p.mutex);
  p.count = 0;
  p.archive.clear();
  p.active = false;
  p.overflow = false;
  p.capturing = false;
  p.recording = true;
  p.next_scope_id = 1;
  p.last_layer = -1;
  if (p.ensure_epoch() == cudaSuccess) cudaEventRecord(p.epoch, stream);
}

inline void record_epoch(cudaStream_t stream) {
  if (!enabled()) return;
  Pool& p = pool();
  std::lock_guard<std::mutex> lock(p.mutex);
  if (p.ensure_epoch() == cudaSuccess) cudaEventRecord(p.epoch, stream);
}

inline int parse_layer(const char* name) {
  if (name == nullptr || name[0] == '\0') return -1;
  const char* blk = std::strstr(name, "blk.");
  if (blk != nullptr) return std::atoi(blk + 4);
  const char* dash = std::strrchr(name, '-');
  if (dash == nullptr || dash[1] < '0' || dash[1] > '9') return -1;
  char* end = nullptr;
  const long value = std::strtol(dash + 1, &end, 10);
  if (end == dash + 1) return -1;
  if (*end != '\0' && *end != ' ' && *end != '(' && *end != '[') return -1;
  return static_cast<int>(value);
}

inline int parse_layer_from_tensor(const ggml_tensor* node) {
  if (node == nullptr) return -1;
  int layer = parse_layer(node->name);
  if (layer >= 0) return layer;
  for (int src = 0; src < 2; ++src) {
    if (node->src[src] == nullptr) continue;
    layer = parse_layer(node->src[src]->name);
    if (layer >= 0) return layer;
  }
  return -1;
}

inline bool name_prefix(const char* name, const char* prefix) {
  if (name == nullptr || prefix == nullptr) return false;
  const std::size_t n = std::strlen(prefix);
  return std::strncmp(name, prefix, n) == 0;
}

inline const char* infer_role(const char* name, const char* op, int fused_count,
                              const char* weight = nullptr) {
  if (name == nullptr) name = "";
  if (op == nullptr) op = "";
  if (weight == nullptr) weight = "";
  char hay[512]{};
  std::snprintf(hay, sizeof(hay), "%s %s %s", name, weight, op);
  if (std::strstr(hay, "ffn_gate") != nullptr && fused_count > 1) {
    return "ffn_gate_up_glu";
  }
  if (std::strstr(hay, "ffn_gate") != nullptr) return "ffn_gate";
  if (std::strstr(hay, "ffn_up") != nullptr) return "ffn_up";
  if (std::strstr(hay, "ffn_out") != nullptr ||
      std::strstr(hay, "ffn_down") != nullptr) {
    return "ffn_down";
  }
  if (std::strcmp(op, "GLU") == 0) return "ffn_glu";
  if (std::strstr(name, "ssm_out") != nullptr ||
      std::strstr(weight, "ssm_out") != nullptr ||
      std::strstr(name, "linear_attn_out") != nullptr ||
      (name[0] == 'z' && name[1] == '-')) {
    return "q8_gdn_output";
  }
  if (std::strcmp(op, "FLASH_ATTN_EXT") == 0 ||
      std::strcmp(op, "FLASH_ATTN") == 0) {
    return "attention_core";
  }
  if (std::strcmp(op, "ROPE") == 0) return "attention_query_prep";
  if (name_prefix(name, "attn_output") ||
      std::strstr(weight, "attn_output.weight") != nullptr) {
    return "q6_attention_output";
  }
  if (std::strcmp(op, "SSM_CONV") == 0) return "gdn_conv";
  if (std::strcmp(op, "GATED_DELTA_NET") == 0) return "gdn_recurrence";
  if (std::strcmp(op, "L2_NORM") == 0) return "gdn_norm";
  if (std::strstr(hay, "attn_qkv") != nullptr ||
      std::strstr(name, "Qcur") != nullptr ||
      std::strstr(name, "Kcur") != nullptr ||
      std::strstr(name, "Vcur") != nullptr ||
      std::strstr(weight, "attn_q.weight") != nullptr ||
      std::strstr(weight, "attn_k.weight") != nullptr ||
      std::strstr(weight, "attn_v.weight") != nullptr ||
      std::strstr(weight, "ssm_alpha") != nullptr ||
      std::strstr(weight, "ssm_beta") != nullptr ||
      std::strstr(weight, "attn_gate.weight") != nullptr) {
    return "q8_input_projections";
  }
  if (std::strstr(hay, "token_embd") != nullptr) return "embedding";
  if (std::strstr(hay, "output_norm") != nullptr) return "logits_norm";
  if (std::strcmp(name, "output") == 0 ||
      std::strstr(weight, "output.weight") != nullptr ||
      std::strstr(name, "result_output") != nullptr) {
    return "logits_projection";
  }
  if (std::strcmp(op, "CPY") == 0 || std::strcmp(op, "CONT") == 0) return "copy";
  if (std::strstr(hay, "conv_state") != nullptr ||
      std::strcmp(op, "SET_ROWS") == 0) {
    return "state_commit";
  }
  if (std::strcmp(op, "ADD") == 0 || std::strcmp(op, "UNARY") == 0 ||
      std::strcmp(op, "RMS_NORM") == 0 || std::strcmp(op, "MUL") == 0 ||
      std::strcmp(op, "SCALE") == 0) {
    return "pointwise";
  }
  if (std::strstr(op, "QUANT") != nullptr) return "activation_quant";
  if (std::strcmp(op, "MUL_MAT") == 0) return "unknown";
  return op[0] != '\0' ? op : "unknown";
}

inline bool begin_op(cudaStream_t stream, int stream_index) {
  if (!enabled()) return false;
  Pool& p = pool();
  std::lock_guard<std::mutex> lock(p.mutex);
  if (!p.recording) return false;
  if (p.active) return false;
  if (p.count >= kPool) {
    p.overflow = true;
    return false;
  }
  if (p.ensure_slot(p.count) != cudaSuccess) {
    p.overflow = true;
    return false;
  }
  if (cudaEventRecord(p.start[p.count], stream) != cudaSuccess) return false;
  p.streams[p.count] = stream;
  Record& rec = p.meta[p.count];
  rec = {};
  copy_str(rec.engine, sizeof(rec.engine), "llama");
  copy_str(rec.phase, sizeof(rec.phase),
           p.phase[0] != '\0' ? p.phase : "decode");
  rec.sequence_position = p.sequence_position;
  rec.token_position = p.token_position;
  rec.stream = reinterpret_cast<unsigned long long>(stream);
  rec.stream_index = stream_index;
  rec.start_event_id = static_cast<int>(p.archive.size() + p.count) * 2;
  rec.end_event_id = rec.start_event_id + 1;
  rec.scope_id = p.next_scope_id++;
  rec.parent_scope_id = p.token_scope_id;
  copy_str(rec.graph_mode, sizeof(rec.graph_mode), graph_mode_label());
  copy_str(rec.launch_family, sizeof(rec.launch_family),
           p.launch_family[0] != '\0' ? p.launch_family : "kernel");
  p.active = true;
  p.launch_family[0] = '\0';
  return true;
}

inline void fill_tensor(Record* rec, const ggml_tensor* node, int fused_skip,
                        const ggml_cgraph* cgraph, int index) {
  if (rec == nullptr || node == nullptr) return;
  const char* name = node->name;
  const char* weight = nullptr;
  if (node->src[0] != nullptr && node->src[0]->name[0] != '\0') {
    weight = node->src[0]->name;
  }
  rec->layer = parse_layer_from_tensor(node);
  Pool& live = pool();
  if (rec->layer < 0) rec->layer = live.last_layer;
  if (rec->layer >= 0) live.last_layer = rec->layer;
  copy_str(rec->tensor_name, sizeof(rec->tensor_name), name);
  copy_str(rec->weight_name, sizeof(rec->weight_name), weight);
  copy_str(rec->ggml_op, sizeof(rec->ggml_op), ggml_op_name(node->op));
  copy_str(rec->tensor_type, sizeof(rec->tensor_type), ggml_type_name(node->type));
  rec->m = static_cast<int>(node->ne[1]);
  rec->n = static_cast<int>(node->ne[0]);
  rec->k = node->src[0] != nullptr ? static_cast<int>(node->src[0]->ne[0]) : 0;
  rec->strides[0] = static_cast<int64_t>(node->nb[0]);
  rec->strides[1] = static_cast<int64_t>(node->nb[1]);
  rec->strides[2] = static_cast<int64_t>(node->nb[2]);
  rec->strides[3] = static_cast<int64_t>(node->nb[3]);
  rec->fused_member_count = fused_skip + 1;
  if (fused_skip > 0) {
    if ((name != nullptr && std::strstr(name, "ffn_gate") != nullptr) ||
        (weight != nullptr && std::strstr(weight, "ffn_gate") != nullptr)) {
      copy_str(rec->fused_member_ids, sizeof(rec->fused_member_ids),
               "ffn_gate,ffn_up,ffn_glu");
      rec->fused_member_count = 3;
    } else {
      std::snprintf(rec->fused_member_ids, sizeof(rec->fused_member_ids),
                    "fused_%d_ops", rec->fused_member_count);
    }
    copy_str(rec->attribution_role, sizeof(rec->attribution_role), "enclosing");
  } else {
    copy_str(rec->attribution_role, sizeof(rec->attribution_role), "member");
  }
  (void)cgraph;
  (void)index;
  copy_str(rec->role, sizeof(rec->role),
           infer_role(name, ggml_op_name(node->op), rec->fused_member_count,
                      weight));
}

inline void end_op(const ggml_tensor* node, const ggml_cgraph* cgraph, int index,
                   int fused_skip) {
  if (!enabled()) return;
  Pool& p = pool();
  std::lock_guard<std::mutex> lock(p.mutex);
  if (!p.active || p.count >= kPool) return;
  cudaEventRecord(p.stop[p.count], p.streams[p.count]);
  fill_tensor(&p.meta[p.count], node, fused_skip, cgraph, index);
#ifdef QW38_OPT060_NVTX
  nvtxEventAttributes_t attr{};
  attr.version = NVTX_VERSION;
  attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
  attr.message.ascii = p.meta[p.count].role;
  attr.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
  attr.payload.ullValue = static_cast<uint64_t>(index);
  nvtxRangePushEx(&attr);
  nvtxRangePop();
#endif
  ++p.count;
  p.active = false;
}

inline void end_graph() {
  if (!enabled()) return;
  Pool& p = pool();
  std::lock_guard<std::mutex> lock(p.mutex);
  if (!p.active || p.count >= kPool) return;
  cudaEventRecord(p.stop[p.count], p.streams[p.count]);
  Record& rec = p.meta[p.count];
  copy_str(rec.role, sizeof(rec.role), "cuda_graph_launch");
  copy_str(rec.launch_family, sizeof(rec.launch_family), "cuda_graph");
  copy_str(rec.attribution_role, sizeof(rec.attribution_role), "enclosing");
  copy_str(rec.graph_mode, sizeof(rec.graph_mode), "cuda_graph");
  rec.fused_member_count = 0;
  ++p.count;
  p.active = false;
}

inline void note_graph_launch(cudaStream_t stream) {
  if (!begin_op(stream, 0)) return;
  end_graph();
}

inline void resolve_locked(Pool& p) {
  for (std::size_t i = 0; i < p.count; ++i) {
    cudaEventSynchronize(p.stop[i]);
    float elapsed = 0.0F;
    float start_ms = 0.0F;
    float end_ms = 0.0F;
    cudaEventElapsedTime(&elapsed, p.start[i], p.stop[i]);
    if (p.epoch != nullptr) {
      cudaEventElapsedTime(&start_ms, p.epoch, p.start[i]);
      cudaEventElapsedTime(&end_ms, p.epoch, p.stop[i]);
    }
    p.meta[i].start_ms = start_ms;
    p.meta[i].end_ms = end_ms;
    p.meta[i].complete_work_ms = elapsed;
    p.meta[i].attributed = true;
    p.meta[i].pool_overflow = p.overflow;
  }
}

inline void resolve() {
  if (!enabled()) return;
  Pool& p = pool();
  std::lock_guard<std::mutex> lock(p.mutex);
  resolve_locked(p);
}

inline void drain() {
  if (!enabled()) return;
  Pool& p = pool();
  std::lock_guard<std::mutex> lock(p.mutex);
  if (!p.recording || p.count == 0) return;
  resolve_locked(p);
  for (std::size_t i = 0; i < p.count; ++i) {
    p.archive.push_back(p.meta[i]);
  }
  p.count = 0;
  p.active = false;
}

inline std::size_t count() { return pool().archive.size() + pool().count; }

inline bool overflow() { return pool().overflow; }

inline const Record* records() { return pool().meta.data(); }

inline void write_record_json(FILE* out, const Record& rec, bool last) {
  std::fprintf(
      out,
      "{\"engine\":\"%s\",\"phase\":\"%s\",\"sequence_position\":%d,"
      "\"token_position\":%d,\"layer\":%d,\"role\":\"%s\","
      "\"tensor_name\":\"%s\",\"weight_name\":\"%s\",\"op\":\"%s\","
      "\"tensor_type\":\"%s\","
      "\"m\":%d,\"n\":%d,"
      "\"k\":%d,\"strides\":[%lld,%lld,%lld,%lld],\"stream\":%llu,"
      "\"stream_index\":%d,\"launch_family\":\"%s\","
      "\"fused_member_ids\":\"%s\",\"fused_member_count\":%d,"
      "\"start_event_id\":%d,\"end_event_id\":%d,\"scope_id\":%d,"
      "\"parent_scope_id\":%d,\"graph_mode\":\"%s\","
      "\"attribution_role\":\"%s\",\"start_ms\":%.9g,\"end_ms\":%.9g,"
      "\"complete_work_ms\":%.9g,\"attributed\":%s,\"pool_overflow\":%s}%s",
      rec.engine, rec.phase, rec.sequence_position, rec.token_position,
      rec.layer, rec.role, rec.tensor_name, rec.weight_name, rec.ggml_op,
      rec.tensor_type,
      rec.m, rec.n,
      rec.k, static_cast<long long>(rec.strides[0]),
      static_cast<long long>(rec.strides[1]),
      static_cast<long long>(rec.strides[2]),
      static_cast<long long>(rec.strides[3]), rec.stream, rec.stream_index,
      rec.launch_family, rec.fused_member_ids, rec.fused_member_count,
      rec.start_event_id, rec.end_event_id, rec.scope_id, rec.parent_scope_id,
      rec.graph_mode, rec.attribution_role, rec.start_ms, rec.end_ms,
      rec.complete_work_ms, rec.attributed ? "true" : "false",
      rec.pool_overflow ? "true" : "false", last ? "" : ",");
}

inline void dump_json(FILE* out) {
  if (out == nullptr) return;
  resolve();
  Pool& p = pool();
  const std::size_t total = p.archive.size() + p.count;
  std::fprintf(out, "{\"schema_version\":1,\"task\":\"OPT-099\",\"engine\":\"llama\",");
  std::fprintf(
      out,
      "\"graph_mode\":\"%s\",\"pool_overflow\":%s,\"recording\":%s,"
      "\"measured_window\":true,\"count\":%zu,\"records\":[",
      graph_mode_label(), p.overflow ? "true" : "false",
      p.recording ? "true" : "false", total);
  for (std::size_t i = 0; i < p.archive.size(); ++i) {
    write_record_json(out, p.archive[i], i + 1 == total);
  }
  for (std::size_t i = 0; i < p.count; ++i) {
    write_record_json(out, p.meta[i], p.archive.size() + i + 1 == total);
  }
  std::fprintf(out, "]}\n");
}

}  // namespace qw38_opt060

#define QW38_OPT060_SET_LAUNCH_FAMILY(name) \
  qw38_opt060::set_launch_family(name)
