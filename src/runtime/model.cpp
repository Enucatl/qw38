#include "runtime/model.hpp"

#include "cuda/stream.hpp"
#include "cuda/upload.hpp"
#include "format/constants.hpp"
#include "runtime/sizes.hpp"

#include <new>
#include <stdexcept>
#include <unordered_map>

namespace qw38::runtime {
namespace {

qw38::format::ArithmeticDtype dtype_for_storage(
    qw38::format::StorageClass storage) noexcept {
  switch (storage) {
    case qw38::format::StorageClass::Fp32:
      return qw38::format::ArithmeticDtype::Fp32;
    case qw38::format::StorageClass::Bf16:
    case qw38::format::StorageClass::Int4Grouped:
    case qw38::format::StorageClass::Int8Grouped:
      return qw38::format::ArithmeticDtype::Bf16;
  }
  return qw38::format::ArithmeticDtype::Bf16;
}

ConstTensorView make_payload_view(qw38::format::TensorRecord const& rec,
                                  void const* ptr) {
  ConstTensorView v{};
  v.pointer = ptr;
  v.dtype = dtype_for_storage(rec.storage);
  v.layout = rec.layout;
  v.storage = rec.storage;
  v.space = MemorySpace::Device;
  v.rank = rec.shape.rank;
  v.extent = rec.shape.logical;
  return v;
}

ConstTensorView make_scale_view(qw38::format::TensorRecord const& rec,
                                void const* ptr) {
  ConstTensorView v{};
  v.pointer = ptr;
  v.dtype = qw38::format::ArithmeticDtype::Fp16;
  v.layout = rec.layout;
  v.storage = rec.storage;
  v.space = MemorySpace::Device;
  v.rank = 1;
  v.extent[0] = rec.scales.length / qw38::format::kFp16Size;
  return v;
}

bool is_alias(qw38::format::ArtifactSchema const& schema, std::uint32_t id) {
  for (auto const& b : schema.shared_bindings) {
    if (b.alias_tensor_id == id) {
      return true;
    }
  }
  return false;
}

std::uint32_t owner_id(qw38::format::ArtifactSchema const& schema,
                       std::uint32_t id) {
  for (auto const& b : schema.shared_bindings) {
    if (b.alias_tensor_id == id) {
      return b.owner_tensor_id;
    }
  }
  return id;
}

qw38::format::TensorRecord const* find_id(
    qw38::format::ArtifactSchema const& schema, std::uint32_t id) {
  for (auto const& t : schema.tensors) {
    if (t.tensor_id == id) {
      return &t;
    }
  }
  return nullptr;
}

}  // namespace

qw38::format::TensorRecord const* Model::find_tensor(
    std::string_view logical_name) const noexcept {
  for (auto const& t : schema_.tensors) {
    if (t.logical_name == logical_name) {
      return &t;
    }
  }
  return nullptr;
}

std::expected<std::uint32_t, Error> resolve_semantic_tensor(
    qw38::format::ArtifactSchema const& schema, std::string_view logical_name,
    qw38::format::SemanticNodeKind kind, qw38::format::TensorRole role,
    std::uint32_t layer) {
  if (layer >= kLanguageLayers) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "layer",
                                      "semantic layer index is invalid"));
  }
  qw38::format::TensorRecord const* record = nullptr;
  for (auto const& tensor : schema.tensors) {
    if (tensor.logical_name == logical_name) {
      if (record != nullptr) {
        return std::unexpected(make_error(ErrorCode::MalformedArtifact,
                                          "tensor", "duplicate logical name"));
      }
      record = &tensor;
    }
  }
  if (record == nullptr) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "tensor", "required semantic tensor is missing"));
  }
  auto const instance = 1u + 2u * layer +
                        static_cast<std::uint32_t>(kind == qw38::format::SemanticNodeKind::Mlp);
  std::uint32_t matches = 0;
  for (auto const& binding : schema.graph_bindings) {
    if (binding.tensor_id != record->tensor_id) continue;
    if (binding.instance_id == instance && binding.kind == kind &&
        binding.role == role && binding.layer_index == layer) {
      ++matches;
    } else if (record->logical_name != "rope.inv_freq" ||
               binding.kind != qw38::format::SemanticNodeKind::GatedAttention ||
               binding.role != qw38::format::TensorRole::VectorWeight ||
               binding.layer_index >= kLanguageLayers ||
               (binding.instance_id != 1u + 2u * binding.layer_index &&
                !(binding.instance_id == 132u && binding.layer_index == 0u))) {
      return std::unexpected(make_error(ErrorCode::MalformedArtifact,
                                        "graph_bindings", "tensor has a wrong semantic binding"));
    }
  }
  if (matches != 1) {
    return std::unexpected(make_error(ErrorCode::MalformedArtifact,
                                      "graph_bindings", "required binding is missing or duplicated"));
  }
  return record->tensor_id;
}

std::expected<std::uint32_t, Error> Model::resolve_tensor(
    std::string_view logical_name, qw38::format::SemanticNodeKind kind,
    qw38::format::TensorRole role, std::uint32_t layer) const {
  return resolve_semantic_tensor(schema_, logical_name, kind, role, layer);
}

std::expected<ConstTensorView, Error> Model::payload(
    std::uint32_t tensor_id) const {
  for (auto const& u : uploaded_) {
    if (u.tensor_id == tensor_id) {
      if (u.payload.pointer == nullptr) {
        return std::unexpected(make_error(ErrorCode::InvalidArgument, "payload",
                                          "tensor has no payload"));
      }
      return u.payload;
    }
  }
  return std::unexpected(make_error(ErrorCode::InvalidArgument, "tensor",
                                    "tensor ID is not uploaded"));
}

std::expected<ConstTensorView, Error> Model::scales(
    std::uint32_t tensor_id) const {
  for (auto const& u : uploaded_) {
    if (u.tensor_id == tensor_id) {
      if (u.scales.pointer == nullptr) {
        return std::unexpected(make_error(ErrorCode::InvalidArgument, "scales",
                                          "tensor has no scales"));
      }
      return u.scales;
    }
  }
  return std::unexpected(make_error(ErrorCode::InvalidArgument, "tensor",
                                    "tensor ID is not uploaded"));
}

std::expected<ConstTensorView, Error> Model::payload(
    std::string_view logical_name) const {
  auto const* rec = find_tensor(logical_name);
  if (rec == nullptr) {
    return std::unexpected(
        make_error(ErrorCode::InvalidArgument, "tensor", "tensor not found"));
  }
  for (auto const& u : uploaded_) {
    if (u.tensor_id == rec->tensor_id) {
      if (u.payload.pointer == nullptr) {
        return std::unexpected(make_error(ErrorCode::InvalidArgument, "payload",
                                          "tensor has no payload"));
      }
      return u.payload;
    }
  }
  return std::unexpected(
      make_error(ErrorCode::Internal, "payload", "uploaded tensor missing"));
}

std::expected<ConstTensorView, Error> Model::scales(
    std::string_view logical_name) const {
  auto const* rec = find_tensor(logical_name);
  if (rec == nullptr) {
    return std::unexpected(
        make_error(ErrorCode::InvalidArgument, "tensor", "tensor not found"));
  }
  for (auto const& u : uploaded_) {
    if (u.tensor_id == rec->tensor_id) {
      if (u.scales.pointer == nullptr) {
        return std::unexpected(make_error(ErrorCode::InvalidArgument, "scales",
                                          "tensor has no scales"));
      }
      return u.scales;
    }
  }
  return std::unexpected(
      make_error(ErrorCode::Internal, "scales", "uploaded tensor missing"));
}

std::expected<Model, Error> Model::upload(qw38::format::Artifact const& artifact,
                                          qw38::cuda::Stream const& stream,
                                          std::optional<DiagnosticWeights> selection,
                                          std::uint32_t layer) {
  try {
  Model model;
  model.schema_ = artifact.schema();
  model.device_ = stream.device();

  // Repeat the complete host-side compatibility preflight at the allocation
  // boundary so no future Artifact construction path can bypass it.
  if (auto st = qw38::format::validate_schema(model.schema_); !st) {
    return std::unexpected(from_format(st.error()));
  }
  if (auto st = require_language_state(model.state()); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = require_language_scratch(model.scratch()); !st) {
    return std::unexpected(st.error());
  }
  if (selection == DiagnosticWeights::Layer && layer >= kLanguageLayers) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "layer",
                                      "diagnostic layer must be < 64"));
  }
  std::string const layer_prefix = selection == DiagnosticWeights::Layer
      ? "model.language_model.layers." + std::to_string(layer) + "." : "";
  auto selected = [&](qw38::format::TensorRecord const& rec) {
    if (!selection) return true;
    switch (*selection) {
      case DiagnosticWeights::Input:
        return rec.logical_name == "model.language_model.embed_tokens.weight";
      case DiagnosticWeights::Layer:
        return rec.logical_name.starts_with(layer_prefix) ||
               rec.logical_name == "rope.inv_freq";
      case DiagnosticWeights::Output:
        return rec.logical_name == "model.language_model.norm.weight" ||
               rec.logical_name == "lm_head.weight";
    }
    return false;
  };

  // Host-side span checks happen before any device allocation.
  for (auto const& rec : model.schema_.tensors) {
    if (!selected(rec) || is_alias(model.schema_, rec.tensor_id)) {
      continue;
    }
    if (!rec.payload.empty()) {
      auto host = artifact.span(rec.logical_name, qw38::format::SpanKind::Payload);
      if (!host) {
        return std::unexpected(from_format(host.error()));
      }
    }
    if (!rec.scales.empty()) {
      auto host = artifact.span(rec.logical_name, qw38::format::SpanKind::Scales);
      if (!host) {
        return std::unexpected(from_format(host.error()));
      }
    }
  }

  struct SpanKey {
    std::uint64_t offset{};
    std::uint64_t length{};
    bool operator==(SpanKey const&) const = default;
  };
  struct SpanKeyHash {
    std::size_t operator()(SpanKey const& k) const noexcept {
      return static_cast<std::size_t>(k.offset ^ (k.length << 1));
    }
  };
  std::unordered_map<SpanKey, std::size_t, SpanKeyHash> buffer_index;

  auto upload_named = [&](std::string_view logical_name,
                          qw38::format::SpanKind kind,
                          qw38::format::ByteSpan span)
      -> std::expected<void*, Error> {
    if (span.empty()) {
      return nullptr;
    }
    SpanKey key{span.offset, span.length};
    if (auto it = buffer_index.find(key); it != buffer_index.end()) {
      return model.buffers_[it->second].data();
    }
    auto host = artifact.span(logical_name, kind);
    if (!host) {
      return std::unexpected(from_format(host.error()));
    }
    auto buf = qw38::cuda::upload(*host, stream);
    if (!buf) {
      return std::unexpected(from_cuda(buf.error()));
    }
    void* ptr = buf->data();
    buffer_index.emplace(key, model.buffers_.size());
    model.device_bytes_ += buf->bytes();
    model.buffers_.push_back(std::move(*buf));
    return ptr;
  };

  for (auto const& rec : model.schema_.tensors) {
    if (!selected(rec) || is_alias(model.schema_, rec.tensor_id)) {
      continue;
    }
    if (auto st = upload_named(rec.logical_name, qw38::format::SpanKind::Payload,
                               rec.payload);
        !st) {
      return std::unexpected(st.error());
    }
    if (auto st = upload_named(rec.logical_name, qw38::format::SpanKind::Scales,
                               rec.scales);
        !st) {
      return std::unexpected(st.error());
    }
  }

  model.uploaded_.reserve(model.schema_.tensors.size());
  for (auto const& rec : model.schema_.tensors) {
    if (!selected(rec)) continue;
    auto const oid = owner_id(model.schema_, rec.tensor_id);
    auto const* owner = find_id(model.schema_, oid);
    if (owner == nullptr) {
      return std::unexpected(make_error(ErrorCode::MalformedArtifact, "tensor",
                                        "shared binding owner missing"));
    }
    UploadedTensor u;
    u.tensor_id = rec.tensor_id;
    if (!owner->payload.empty()) {
      auto ptr = upload_named(owner->logical_name, qw38::format::SpanKind::Payload,
                              owner->payload);
      if (!ptr) {
        return std::unexpected(ptr.error());
      }
      u.payload = make_payload_view(rec, *ptr);
    }
    if (!owner->scales.empty()) {
      auto ptr = upload_named(owner->logical_name, qw38::format::SpanKind::Scales,
                              owner->scales);
      if (!ptr) {
        return std::unexpected(ptr.error());
      }
      u.scales = make_scale_view(rec, *ptr);
    }
    model.uploaded_.push_back(u);
  }

  if (auto st = stream.sync(); !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  return model;
  } catch (std::bad_alloc const&) {
    return std::unexpected(make_error(ErrorCode::AllocationFailed, "model.upload",
                                      "host allocation failed"));
  } catch (std::length_error const&) {
    return std::unexpected(make_error(ErrorCode::AllocationFailed, "model.upload",
                                      "host allocation size is invalid"));
  }
}

}  // namespace qw38::runtime
