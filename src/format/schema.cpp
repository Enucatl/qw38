#include "format/schema.hpp"

#include <iterator>
#include <limits>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace qw38::format {

std::expected<std::span<std::uint64_t const>, FormatError>
TensorShape::logical_dims(std::uint64_t offset,
                          std::string_view field) const noexcept {
  if (rank == 0 || rank > kMaxRank) {
    return std::unexpected(make_error(FormatErrorCode::InvalidShape, offset,
                                      field, "rank must be 1..8"));
  }
  return std::span<std::uint64_t const>{logical.data(), rank};
}

std::expected<std::span<std::uint64_t const>, FormatError>
TensorShape::padded_dims(std::uint64_t offset,
                         std::string_view field) const noexcept {
  if (rank == 0 || rank > kMaxRank) {
    return std::unexpected(make_error(FormatErrorCode::InvalidShape, offset,
                                      field, "rank must be 1..8"));
  }
  return std::span<std::uint64_t const>{padded.data(), rank};
}

namespace {

constexpr std::uint16_t kFlagPopulatedDistinct = 0x0001;
constexpr std::uint16_t kFlagLivePayload = 0x0002;

template <typename Count>
std::expected<void, FormatError> preflight_count(
    ByteReader const& r, Count count, std::size_t minimum_record_bytes,
    Count maximum_count, std::string_view field) {
  auto const count_offset = r.offset() - sizeof(Count);
  if (count > maximum_count) {
    return std::unexpected(make_error(
        FormatErrorCode::ResourceLimitExceeded, count_offset, field,
        "record count exceeds the V0 limit"));
  }
  if (static_cast<std::size_t>(count) >
      r.remaining() / minimum_record_bytes) {
    return std::unexpected(make_error(
        FormatErrorCode::Truncated, count_offset, field,
        "remaining manifest bytes cannot contain the declared records"));
  }
  return {};
}

std::expected<std::size_t, FormatError> to_size(std::uint64_t n,
                                                std::uint64_t offset,
                                                std::string_view field) {
  if (n > std::numeric_limits<std::size_t>::max()) {
    return std::unexpected(make_error(FormatErrorCode::Overflow, offset, field,
                                      "encoded size exceeds size_t"));
  }
  return static_cast<std::size_t>(n);
}

class SizeAcc {
 public:
  std::expected<void, FormatError> add(std::uint64_t n, std::uint64_t offset,
                                       std::string_view field) {
    auto const next = checked_add(n_, n, offset, field);
    if (!next) {
      return std::unexpected(next.error());
    }
    n_ = *next;
    return {};
  }

  std::expected<std::size_t, FormatError> finish(std::uint64_t offset,
                                                 std::string_view field) const {
    return to_size(n_, offset, field);
  }

  std::uint64_t get() const noexcept { return n_; }

 private:
  std::uint64_t n_{0};
};

std::expected<void, FormatError> validate_shape(TensorShape const& shape,
                                                std::uint64_t offset,
                                                std::string_view field) {
  if (shape.rank == 0 || shape.rank > kMaxRank) {
    return std::unexpected(make_error(FormatErrorCode::InvalidShape, offset,
                                      field, "rank must be 1..8"));
  }
  for (std::uint8_t i = 0; i < shape.rank; ++i) {
    if (shape.logical[i] == 0) {
      return std::unexpected(make_error(FormatErrorCode::InvalidShape, offset,
                                        field, "logical extent must be > 0"));
    }
    if (shape.padded[i] < shape.logical[i]) {
      return std::unexpected(make_error(FormatErrorCode::InvalidShape, offset,
                                        field, "padded extent < logical"));
    }
  }
  for (std::uint8_t i = shape.rank; i < kMaxRank; ++i) {
    if (shape.logical[i] != 0 || shape.padded[i] != 0) {
      return std::unexpected(make_error(FormatErrorCode::InvalidShape, offset,
                                        field, "unused dims must be zero"));
    }
  }
  return {};
}

std::expected<void, FormatError> write_hash(ByteWriter& w, Hash256 const& hash,
                                            std::string_view field) {
  std::array<std::byte, 32> raw{};
  for (std::size_t i = 0; i < hash.bytes.size(); ++i) {
    raw[i] = static_cast<std::byte>(hash.bytes[i]);
  }
  return w.bytes(raw, field);
}

std::expected<Hash256, FormatError> read_hash(ByteReader& r,
                                              std::string_view field) {
  std::array<std::byte, 32> raw{};
  if (auto st = r.bytes(raw, field); !st) {
    return std::unexpected(st.error());
  }
  Hash256 hash{};
  for (std::size_t i = 0; i < hash.bytes.size(); ++i) {
    hash.bytes[i] = static_cast<std::uint8_t>(raw[i]);
  }
  return hash;
}

std::expected<void, FormatError> write_span(ByteWriter& w, ByteSpan const& span,
                                            std::string_view offset_field,
                                            std::string_view length_field) {
  if (auto st = w.u64(span.offset, offset_field); !st) {
    return st;
  }
  return w.u64(span.length, length_field);
}

std::expected<ByteSpan, FormatError> read_span(ByteReader& r,
                                               std::string_view offset_field,
                                               std::string_view length_field) {
  auto const offset = r.u64(offset_field);
  if (!offset) {
    return std::unexpected(offset.error());
  }
  auto const length = r.u64(length_field);
  if (!length) {
    return std::unexpected(length.error());
  }
  return ByteSpan{.offset = *offset, .length = *length};
}

std::expected<std::uint64_t, FormatError> shape_encoded_bytes(
    TensorShape const& shape, std::uint64_t offset) {
  if (auto st = validate_shape(shape, offset, "shape"); !st) {
    return std::unexpected(st.error());
  }
  auto logical = shape.logical_dims(offset, "shape.logical");
  auto padded = shape.padded_dims(offset, "shape.padded");
  if (!logical || !padded) {
    return std::unexpected(!logical ? logical.error() : padded.error());
  }
  if (auto product = checked_product(*logical, offset, "shape.logical");
      !product) {
    return std::unexpected(product.error());
  }
  if (auto product = checked_product(*padded, offset, "shape.padded");
      !product) {
    return std::unexpected(product.error());
  }
  SizeAcc acc;
  if (auto st = acc.add(8, offset, "shape"); !st) {
    return std::unexpected(st.error());
  }
  auto const dims = checked_mul(shape.rank, 16, offset, "shape.dims");
  if (!dims) {
    return std::unexpected(dims.error());
  }
  if (auto st = acc.add(*dims, offset, "shape.dims"); !st) {
    return std::unexpected(st.error());
  }
  return acc.get();
}

std::expected<void, FormatError> write_shape(ByteWriter& w,
                                             TensorShape const& shape,
                                             std::string_view field) {
  if (auto st = validate_shape(shape, w.offset(), field); !st) {
    return st;
  }
  if (auto st = w.u8(shape.rank, field); !st) {
    return st;
  }
  if (auto st = w.zeros(7, field); !st) {
    return st;
  }
  for (std::uint8_t i = 0; i < shape.rank; ++i) {
    if (auto st = w.u64(shape.logical[i], field); !st) {
      return st;
    }
    if (auto st = w.u64(shape.padded[i], field); !st) {
      return st;
    }
  }
  return {};
}

std::expected<TensorShape, FormatError> read_shape(ByteReader& r,
                                                   std::string_view field) {
  TensorShape shape{};
  auto const rank = r.u8(field);
  if (!rank) {
    return std::unexpected(rank.error());
  }
  shape.rank = *rank;
  if (auto st = r.expect_zeros(7, field); !st) {
    return std::unexpected(st.error());
  }
  if (shape.rank == 0 || shape.rank > kMaxRank) {
    return std::unexpected(make_error(FormatErrorCode::InvalidShape, r.offset(),
                                      field, "rank must be 1..8"));
  }
  for (std::uint8_t i = 0; i < shape.rank; ++i) {
    auto const logical = r.u64(field);
    if (!logical) {
      return std::unexpected(logical.error());
    }
    auto const padded = r.u64(field);
    if (!padded) {
      return std::unexpected(padded.error());
    }
    shape.logical[i] = *logical;
    shape.padded[i] = *padded;
  }
  return shape;
}

std::expected<void, FormatError> write_mapping(ByteWriter& w,
                                               LogicalPhysicalMapping const& m) {
  if (auto st = w.u16(std::to_underlying(m.kind), "mapping.kind"); !st) {
    return st;
  }
  if (auto st = w.u16(0, "mapping.reserved"); !st) {
    return st;
  }
  if (auto st = w.u32(m.tile_rows, "mapping.tile_rows"); !st) {
    return st;
  }
  if (auto st = w.u32(m.tile_k, "mapping.tile_k"); !st) {
    return st;
  }
  if (auto st = w.u32(m.group_size, "mapping.group_size"); !st) {
    return st;
  }
  return w.u32(m.packed_bytes_per_tile_row, "mapping.packed_bytes_per_tile_row");
}

std::expected<LogicalPhysicalMapping, FormatError> read_mapping(ByteReader& r) {
  LogicalPhysicalMapping m{};
  auto const kind = r.u16("mapping.kind");
  if (!kind) {
    return std::unexpected(kind.error());
  }
  auto decoded = decode_mapping_kind(*kind, r.offset() - 2, "mapping.kind");
  if (!decoded) {
    return std::unexpected(decoded.error());
  }
  m.kind = *decoded;
  if (auto st = r.expect_zeros(2, "mapping.reserved"); !st) {
    return std::unexpected(st.error());
  }
  auto const tile_rows = r.u32("mapping.tile_rows");
  if (!tile_rows) {
    return std::unexpected(tile_rows.error());
  }
  auto const tile_k = r.u32("mapping.tile_k");
  if (!tile_k) {
    return std::unexpected(tile_k.error());
  }
  auto const group = r.u32("mapping.group_size");
  if (!group) {
    return std::unexpected(group.error());
  }
  auto const packed = r.u32("mapping.packed_bytes_per_tile_row");
  if (!packed) {
    return std::unexpected(packed.error());
  }
  m.tile_rows = *tile_rows;
  m.tile_k = *tile_k;
  m.group_size = *group;
  m.packed_bytes_per_tile_row = *packed;
  return m;
}

std::expected<void, FormatError> write_compiler(ByteWriter& w,
                                                CompilerRevision const& rev) {
  if (auto st = w.name(rev.ident, "compiler.ident"); !st) {
    return st;
  }
  if (auto st = w.u32(rev.major, "compiler.major"); !st) {
    return st;
  }
  if (auto st = w.u32(rev.minor, "compiler.minor"); !st) {
    return st;
  }
  if (auto st = w.u32(rev.patch, "compiler.patch"); !st) {
    return st;
  }
  return w.u32(0, "compiler.reserved");
}

std::expected<CompilerRevision, FormatError> read_compiler(ByteReader& r) {
  CompilerRevision rev{};
  auto const ident = r.name("compiler.ident");
  if (!ident) {
    return std::unexpected(ident.error());
  }
  rev.ident = *ident;
  auto const major = r.u32("compiler.major");
  if (!major) {
    return std::unexpected(major.error());
  }
  auto const minor = r.u32("compiler.minor");
  if (!minor) {
    return std::unexpected(minor.error());
  }
  auto const patch = r.u32("compiler.patch");
  if (!patch) {
    return std::unexpected(patch.error());
  }
  if (auto st = r.expect_zeros(4, "compiler.reserved"); !st) {
    return std::unexpected(st.error());
  }
  rev.major = *major;
  rev.minor = *minor;
  rev.patch = *patch;
  return rev;
}

std::expected<void, FormatError> write_precision(
    ByteWriter& w, PrecisionPolicyRecord const& policy) {
  if (auto st = w.u16(std::to_underlying(policy.id), "precision.id"); !st) {
    return st;
  }
  if (policy.bindings.size() > std::numeric_limits<std::uint16_t>::max()) {
    return std::unexpected(make_error(FormatErrorCode::Overflow, w.offset(),
                                      "precision.bindings",
                                      "too many precision bindings"));
  }
  if (auto st = w.u16(static_cast<std::uint16_t>(policy.bindings.size()),
                      "precision.count");
      !st) {
    return st;
  }
  for (auto const& b : policy.bindings) {
    if (auto st = w.u16(std::to_underlying(b.domain), "precision.domain"); !st) {
      return st;
    }
    if (auto st = w.u16(std::to_underlying(b.dtype), "precision.dtype"); !st) {
      return st;
    }
  }
  return {};
}

std::expected<PrecisionPolicyRecord, FormatError> read_precision(ByteReader& r) {
  PrecisionPolicyRecord policy{};
  auto const id = r.u16("precision.id");
  if (!id) {
    return std::unexpected(id.error());
  }
  auto decoded_id =
      decode_precision_policy_id(*id, r.offset() - 2, "precision.id");
  if (!decoded_id) {
    return std::unexpected(decoded_id.error());
  }
  policy.id = *decoded_id;
  auto const count = r.u16("precision.count");
  if (!count) {
    return std::unexpected(count.error());
  }
  if (auto st = preflight_count(r, *count, 4, kMaxPrecisionBindingsV0,
                                "precision.count");
      !st) {
    return std::unexpected(st.error());
  }
  policy.bindings.reserve(*count);
  for (std::uint16_t i = 0; i < *count; ++i) {
    auto const domain = r.u16("precision.domain");
    if (!domain) {
      return std::unexpected(domain.error());
    }
    auto decoded_domain =
        decode_precision_domain(*domain, r.offset() - 2, "precision.domain");
    if (!decoded_domain) {
      return std::unexpected(decoded_domain.error());
    }
    auto const dtype = r.u16("precision.dtype");
    if (!dtype) {
      return std::unexpected(dtype.error());
    }
    auto decoded_dtype =
        decode_arithmetic_dtype(*dtype, r.offset() - 2, "precision.dtype");
    if (!decoded_dtype) {
      return std::unexpected(decoded_dtype.error());
    }
    policy.bindings.push_back(PrecisionBinding{.domain = *decoded_domain,
                                               .dtype = *decoded_dtype});
  }
  return policy;
}

std::expected<std::uint64_t, FormatError> tensor_encoded_bytes(
    TensorRecord const& tensor, std::uint64_t offset) {
  SizeAcc acc;
  if (auto st = acc.add(4, offset, "tensor.id"); !st) {
    return std::unexpected(st.error());
  }
  auto const name_len = checked_add(2, tensor.logical_name.size(), offset,
                                    "tensor.logical_name");
  if (!name_len) {
    return std::unexpected(name_len.error());
  }
  if (auto st = acc.add(*name_len, offset, "tensor.logical_name"); !st) {
    return std::unexpected(st.error());
  }
  auto const shape_n = shape_encoded_bytes(tensor.shape, offset);
  if (!shape_n) {
    return std::unexpected(shape_n.error());
  }
  if (auto st = acc.add(*shape_n, offset, "tensor.shape"); !st) {
    return std::unexpected(st.error());
  }
  // storage, quantizer, layout, reserved (8), mapping (20), payload (16), scales (16)
  if (auto st = acc.add(8 + 20 + 16 + 16, offset, "tensor.fixed"); !st) {
    return std::unexpected(st.error());
  }
  return acc.get();
}

std::expected<void, FormatError> write_tensor(ByteWriter& w,
                                              TensorRecord const& tensor) {
  if (auto st = w.u32(tensor.tensor_id, "tensor.id"); !st) {
    return st;
  }
  if (auto st = w.name(tensor.logical_name, "tensor.logical_name"); !st) {
    return st;
  }
  if (auto st = write_shape(w, tensor.shape, "tensor.shape"); !st) {
    return st;
  }
  if (auto st = w.u16(std::to_underlying(tensor.storage), "tensor.storage");
      !st) {
    return st;
  }
  if (auto st =
          w.u16(std::to_underlying(tensor.quantizer), "tensor.quantizer");
      !st) {
    return st;
  }
  if (auto st = w.u16(std::to_underlying(tensor.layout), "tensor.layout"); !st) {
    return st;
  }
  if (auto st = w.u16(0, "tensor.reserved"); !st) {
    return st;
  }
  if (auto st = write_mapping(w, tensor.mapping); !st) {
    return st;
  }
  if (auto st = write_span(w, tensor.payload, "tensor.payload.offset",
                           "tensor.payload.length");
      !st) {
    return st;
  }
  return write_span(w, tensor.scales, "tensor.scales.offset",
                    "tensor.scales.length");
}

std::expected<TensorRecord, FormatError> read_tensor(ByteReader& r) {
  TensorRecord tensor{};
  auto const id = r.u32("tensor.id");
  if (!id) {
    return std::unexpected(id.error());
  }
  tensor.tensor_id = *id;
  auto const name = r.name("tensor.logical_name");
  if (!name) {
    return std::unexpected(name.error());
  }
  tensor.logical_name = *name;
  auto const shape = read_shape(r, "tensor.shape");
  if (!shape) {
    return std::unexpected(shape.error());
  }
  tensor.shape = *shape;
  auto const storage = r.u16("tensor.storage");
  if (!storage) {
    return std::unexpected(storage.error());
  }
  auto decoded_storage =
      decode_storage_class(*storage, r.offset() - 2, "tensor.storage");
  if (!decoded_storage) {
    return std::unexpected(decoded_storage.error());
  }
  tensor.storage = *decoded_storage;
  auto const quantizer = r.u16("tensor.quantizer");
  if (!quantizer) {
    return std::unexpected(quantizer.error());
  }
  auto decoded_q =
      decode_logical_quantizer(*quantizer, r.offset() - 2, "tensor.quantizer");
  if (!decoded_q) {
    return std::unexpected(decoded_q.error());
  }
  tensor.quantizer = *decoded_q;
  auto const layout = r.u16("tensor.layout");
  if (!layout) {
    return std::unexpected(layout.error());
  }
  auto decoded_layout =
      decode_physical_layout(*layout, r.offset() - 2, "tensor.layout");
  if (!decoded_layout) {
    return std::unexpected(decoded_layout.error());
  }
  tensor.layout = *decoded_layout;
  if (auto st = r.expect_zeros(2, "tensor.reserved"); !st) {
    return std::unexpected(st.error());
  }
  auto const mapping = read_mapping(r);
  if (!mapping) {
    return std::unexpected(mapping.error());
  }
  tensor.mapping = *mapping;
  auto const payload =
      read_span(r, "tensor.payload.offset", "tensor.payload.length");
  if (!payload) {
    return std::unexpected(payload.error());
  }
  tensor.payload = *payload;
  auto const scales =
      read_span(r, "tensor.scales.offset", "tensor.scales.length");
  if (!scales) {
    return std::unexpected(scales.error());
  }
  tensor.scales = *scales;
  return tensor;
}

std::expected<void, FormatError> write_shared(ByteWriter& w,
                                              SharedBinding const& b) {
  if (auto st = w.u32(b.owner_tensor_id, "shared.owner"); !st) {
    return st;
  }
  if (auto st = w.u32(b.alias_tensor_id, "shared.alias"); !st) {
    return st;
  }
  if (auto st = w.u16(std::to_underlying(b.role), "shared.role"); !st) {
    return st;
  }
  return w.u16(0, "shared.reserved");
}

std::expected<SharedBinding, FormatError> read_shared(ByteReader& r) {
  SharedBinding b{};
  auto const owner = r.u32("shared.owner");
  if (!owner) {
    return std::unexpected(owner.error());
  }
  auto const alias = r.u32("shared.alias");
  if (!alias) {
    return std::unexpected(alias.error());
  }
  auto const role = r.u16("shared.role");
  if (!role) {
    return std::unexpected(role.error());
  }
  auto decoded = decode_shared_binding_role(*role, r.offset() - 2, "shared.role");
  if (!decoded) {
    return std::unexpected(decoded.error());
  }
  if (auto st = r.expect_zeros(2, "shared.reserved"); !st) {
    return std::unexpected(st.error());
  }
  b.owner_tensor_id = *owner;
  b.alias_tensor_id = *alias;
  b.role = *decoded;
  return b;
}

std::expected<void, FormatError> write_graph(ByteWriter& w,
                                             GraphBinding const& b) {
  if (auto st = w.u32(b.instance_id, "graph.instance"); !st) {
    return st;
  }
  if (auto st = w.u16(std::to_underlying(b.kind), "graph.kind"); !st) {
    return st;
  }
  if (auto st = w.u16(std::to_underlying(b.role), "graph.role"); !st) {
    return st;
  }
  if (auto st = w.u32(b.layer_index, "graph.layer"); !st) {
    return st;
  }
  return w.u32(b.tensor_id, "graph.tensor");
}

std::expected<GraphBinding, FormatError> read_graph(ByteReader& r) {
  GraphBinding b{};
  auto const instance = r.u32("graph.instance");
  if (!instance) {
    return std::unexpected(instance.error());
  }
  auto const kind = r.u16("graph.kind");
  if (!kind) {
    return std::unexpected(kind.error());
  }
  auto decoded_kind =
      decode_semantic_node(*kind, r.offset() - 2, "graph.kind");
  if (!decoded_kind) {
    return std::unexpected(decoded_kind.error());
  }
  auto const role = r.u16("graph.role");
  if (!role) {
    return std::unexpected(role.error());
  }
  auto decoded_role = decode_tensor_role(*role, r.offset() - 2, "graph.role");
  if (!decoded_role) {
    return std::unexpected(decoded_role.error());
  }
  auto const layer = r.u32("graph.layer");
  if (!layer) {
    return std::unexpected(layer.error());
  }
  auto const tensor = r.u32("graph.tensor");
  if (!tensor) {
    return std::unexpected(tensor.error());
  }
  b.instance_id = *instance;
  b.kind = *decoded_kind;
  b.role = *decoded_role;
  b.layer_index = *layer;
  b.tensor_id = *tensor;
  return b;
}

std::expected<std::uint64_t, FormatError> state_encoded_bytes(
    StateAllocation const& state, std::uint64_t offset) {
  auto const shape_n = shape_encoded_bytes(state.shape_per_layer, offset);
  if (!shape_n) {
    return std::unexpected(shape_n.error());
  }
  return checked_add(48, *shape_n, offset, "state");
}

std::expected<void, FormatError> write_state(ByteWriter& w,
                                             StateAllocation const& s) {
  if (auto st = w.u16(std::to_underlying(s.kind), "state.kind"); !st) {
    return st;
  }
  if (auto st = w.u16(std::to_underlying(s.dtype), "state.dtype"); !st) {
    return st;
  }
  if (auto st = w.u16(std::to_underlying(s.layout), "state.layout"); !st) {
    return st;
  }
  std::uint16_t flags = 0;
  if (s.populated_length_distinct_from_capacity) {
    flags = static_cast<std::uint16_t>(flags | kFlagPopulatedDistinct);
  }
  if (s.live_payload_present) {
    flags = static_cast<std::uint16_t>(flags | kFlagLivePayload);
  }
  if (auto st = w.u16(flags, "state.flags"); !st) {
    return st;
  }
  if (auto st = w.u32(s.layer_count, "state.layer_count"); !st) {
    return st;
  }
  if (auto st = w.u32(s.component_count, "state.component_count"); !st) {
    return st;
  }
  if (auto st = w.u64(s.declared_capacity, "state.declared_capacity"); !st) {
    return st;
  }
  if (auto st = w.u64(s.bytes_per_layer, "state.bytes_per_layer"); !st) {
    return st;
  }
  if (auto st = w.u64(s.bytes_per_token, "state.bytes_per_token"); !st) {
    return st;
  }
  if (auto st = w.u64(s.total_bytes, "state.total_bytes"); !st) {
    return st;
  }
  return write_shape(w, s.shape_per_layer, "state.shape");
}

std::expected<StateAllocation, FormatError> read_state(ByteReader& r) {
  StateAllocation s{};
  auto const kind = r.u16("state.kind");
  if (!kind) {
    return std::unexpected(kind.error());
  }
  auto decoded_kind = decode_state_kind(*kind, r.offset() - 2, "state.kind");
  if (!decoded_kind) {
    return std::unexpected(decoded_kind.error());
  }
  auto const dtype = r.u16("state.dtype");
  if (!dtype) {
    return std::unexpected(dtype.error());
  }
  auto decoded_dtype =
      decode_arithmetic_dtype(*dtype, r.offset() - 2, "state.dtype");
  if (!decoded_dtype) {
    return std::unexpected(decoded_dtype.error());
  }
  auto const layout = r.u16("state.layout");
  if (!layout) {
    return std::unexpected(layout.error());
  }
  auto decoded_layout =
      decode_physical_layout(*layout, r.offset() - 2, "state.layout");
  if (!decoded_layout) {
    return std::unexpected(decoded_layout.error());
  }
  auto const flags = r.u16("state.flags");
  if (!flags) {
    return std::unexpected(flags.error());
  }
  if ((*flags & static_cast<std::uint16_t>(~(kFlagPopulatedDistinct | kFlagLivePayload))) !=
      0) {
    return std::unexpected(make_error(FormatErrorCode::ReservedNonzero,
                                      r.offset() - 2, "state.flags",
                                      "unknown state flags"));
  }
  auto const layers = r.u32("state.layer_count");
  if (!layers) {
    return std::unexpected(layers.error());
  }
  auto const components = r.u32("state.component_count");
  if (!components) {
    return std::unexpected(components.error());
  }
  auto const capacity = r.u64("state.declared_capacity");
  if (!capacity) {
    return std::unexpected(capacity.error());
  }
  auto const bpl = r.u64("state.bytes_per_layer");
  if (!bpl) {
    return std::unexpected(bpl.error());
  }
  auto const bpt = r.u64("state.bytes_per_token");
  if (!bpt) {
    return std::unexpected(bpt.error());
  }
  auto const total = r.u64("state.total_bytes");
  if (!total) {
    return std::unexpected(total.error());
  }
  auto const shape = read_shape(r, "state.shape");
  if (!shape) {
    return std::unexpected(shape.error());
  }
  s.kind = *decoded_kind;
  s.dtype = *decoded_dtype;
  s.layout = *decoded_layout;
  s.populated_length_distinct_from_capacity =
      (*flags & kFlagPopulatedDistinct) != 0;
  s.live_payload_present = (*flags & kFlagLivePayload) != 0;
  s.layer_count = *layers;
  s.component_count = *components;
  s.declared_capacity = *capacity;
  s.bytes_per_layer = *bpl;
  s.bytes_per_token = *bpt;
  s.total_bytes = *total;
  s.shape_per_layer = *shape;
  return s;
}

std::expected<void, FormatError> write_scratch(ByteWriter& w,
                                               ScratchAllocation const& s) {
  if (auto st = w.u16(std::to_underlying(s.kind), "scratch.kind"); !st) {
    return st;
  }
  if (auto st = w.u16(std::to_underlying(s.dtype), "scratch.dtype"); !st) {
    return st;
  }
  if (auto st = w.u64(s.bytes, "scratch.bytes"); !st) {
    return st;
  }
  return w.u32(0, "scratch.reserved");
}

std::expected<ScratchAllocation, FormatError> read_scratch(ByteReader& r) {
  ScratchAllocation s{};
  auto const kind = r.u16("scratch.kind");
  if (!kind) {
    return std::unexpected(kind.error());
  }
  auto decoded_kind = decode_scratch_kind(*kind, r.offset() - 2, "scratch.kind");
  if (!decoded_kind) {
    return std::unexpected(decoded_kind.error());
  }
  auto const dtype = r.u16("scratch.dtype");
  if (!dtype) {
    return std::unexpected(dtype.error());
  }
  auto decoded_dtype =
      decode_arithmetic_dtype(*dtype, r.offset() - 2, "scratch.dtype");
  if (!decoded_dtype) {
    return std::unexpected(decoded_dtype.error());
  }
  auto const bytes = r.u64("scratch.bytes");
  if (!bytes) {
    return std::unexpected(bytes.error());
  }
  if (auto st = r.expect_zeros(4, "scratch.reserved"); !st) {
    return std::unexpected(st.error());
  }
  s.kind = *decoded_kind;
  s.dtype = *decoded_dtype;
  s.bytes = *bytes;
  return s;
}

std::expected<void, FormatError> write_integrity(ByteWriter& w,
                                                 IntegrityRecord const& rec) {
  if (auto st = w.u16(std::to_underlying(rec.kind), "integrity.kind"); !st) {
    return st;
  }
  if (auto st = w.u16(0, "integrity.reserved"); !st) {
    return st;
  }
  if (auto st = w.u32(rec.tensor_id, "integrity.tensor"); !st) {
    return st;
  }
  if (auto st = write_span(w, rec.region, "integrity.offset", "integrity.length");
      !st) {
    return st;
  }
  return write_hash(w, rec.digest, "integrity.digest");
}

std::expected<IntegrityRecord, FormatError> read_integrity(ByteReader& r) {
  IntegrityRecord rec{};
  auto const kind = r.u16("integrity.kind");
  if (!kind) {
    return std::unexpected(kind.error());
  }
  auto decoded = decode_integrity_kind(*kind, r.offset() - 2, "integrity.kind");
  if (!decoded) {
    return std::unexpected(decoded.error());
  }
  if (auto st = r.expect_zeros(2, "integrity.reserved"); !st) {
    return std::unexpected(st.error());
  }
  auto const tensor = r.u32("integrity.tensor");
  if (!tensor) {
    return std::unexpected(tensor.error());
  }
  auto const region = read_span(r, "integrity.offset", "integrity.length");
  if (!region) {
    return std::unexpected(region.error());
  }
  auto const digest = read_hash(r, "integrity.digest");
  if (!digest) {
    return std::unexpected(digest.error());
  }
  rec.kind = *decoded;
  rec.tensor_id = *tensor;
  rec.region = *region;
  rec.digest = *digest;
  return rec;
}

bool tensors_share_payload(TensorRecord const& a,
                           TensorRecord const& b) noexcept {
  return a.shape == b.shape && a.storage == b.storage &&
         a.quantizer == b.quantizer && a.layout == b.layout &&
         a.mapping == b.mapping && a.payload == b.payload &&
         a.scales == b.scales && !a.payload.empty();
}

TensorRecord const* find_tensor(ArtifactSchema const& schema,
                                std::uint32_t id) {
  for (auto const& t : schema.tensors) {
    if (t.tensor_id == id) {
      return &t;
    }
  }
  return nullptr;
}

bool has_graph_role(ArtifactSchema const& schema, std::uint32_t tensor_id,
                    SemanticNodeKind kind, TensorRole role) {
  for (auto const& g : schema.graph_bindings) {
    if (g.tensor_id == tensor_id && g.kind == kind && g.role == role) {
      return true;
    }
  }
  return false;
}

std::expected<void, FormatError> require_shape_rank(
    TensorShape const& shape, std::uint8_t rank, std::uint64_t offset,
    std::string_view field) {
  if (shape.rank != rank) {
    return std::unexpected(make_error(FormatErrorCode::InvalidShape, offset,
                                      field, "unexpected rank"));
  }
  return {};
}

std::expected<void, FormatError> validate_mapping_for_tensor(
    TensorRecord const& tensor, std::uint64_t offset) {
  auto const& m = tensor.mapping;
  if (layout_is_tiled_dense(tensor.layout)) {
    if (m.kind != MappingKind::DenseTileNK) {
      return std::unexpected(make_error(FormatErrorCode::InvalidMapping, offset,
                                        "tensor.mapping.kind",
                                        "tiled layout requires DenseTileNK"));
    }
    if (m.tile_rows != kDenseTileRows || m.tile_k != kDenseTileK) {
      return std::unexpected(make_error(FormatErrorCode::InvalidMapping, offset,
                                        "tensor.mapping.tile",
                                        "tile geometry must be 8 x 256"));
    }
    if (tensor.shape.rank != 2) {
      return std::unexpected(make_error(FormatErrorCode::InvalidShape, offset,
                                        "tensor.shape",
                                        "dense tiled tensors are rank-2 [N,K]"));
    }
    if (tensor.shape.padded[0] % kDenseTileRows != 0 ||
        tensor.shape.padded[1] % kDenseTileK != 0) {
      return std::unexpected(make_error(
          FormatErrorCode::InvalidShape, offset, "tensor.shape.padded",
          "padded N must divide 8 and padded K must divide 256"));
    }
    if (tensor.layout == PhysicalLayoutId::CudaQ4G64V0) {
      if (m.group_size != kQ4GroupSize ||
          m.packed_bytes_per_tile_row != kQ4PackedBytesPerTileRow) {
        return std::unexpected(make_error(
            FormatErrorCode::InvalidMapping, offset, "tensor.mapping",
            "Q4G64 mapping must use group 64 and 128 packed bytes"));
      }
    } else if (tensor.layout == PhysicalLayoutId::CudaQ8G32V0) {
      if (m.group_size != kQ8GroupSize ||
          m.packed_bytes_per_tile_row != kQ8PackedBytesPerTileRow) {
        return std::unexpected(make_error(
            FormatErrorCode::InvalidMapping, offset, "tensor.mapping",
            "Q8G32 mapping must use group 32 and 256 packed bytes"));
      }
    } else if (m.group_size != 0 ||
               m.packed_bytes_per_tile_row != kBf16PackedBytesPerTileRow) {
      return std::unexpected(make_error(
          FormatErrorCode::InvalidMapping, offset, "tensor.mapping",
          "BF16 dense tile mapping must use group 0 and 512 packed bytes"));
    }
    return {};
  }
  if (m.kind != MappingKind::Identity || m.tile_rows != 0 || m.tile_k != 0 ||
      m.group_size != 0 || m.packed_bytes_per_tile_row != 0) {
    return std::unexpected(make_error(FormatErrorCode::InvalidMapping, offset,
                                      "tensor.mapping",
                                      "non-tiled layout requires identity mapping"));
  }
  if (tensor.layout == PhysicalLayoutId::CudaBf16VectorV0 ||
      tensor.layout == PhysicalLayoutId::CudaFp32VectorV0) {
    if (auto st = require_shape_rank(tensor.shape, 1, offset, "tensor.shape");
        !st) {
      return st;
    }
  } else if (auto st =
                 require_shape_rank(tensor.shape, 2, offset, "tensor.shape");
             !st) {
    return st;
  }
  for (std::uint8_t i = 0; i < tensor.shape.rank; ++i) {
    if (tensor.shape.padded[i] != tensor.shape.logical[i]) {
      return std::unexpected(make_error(
          FormatErrorCode::InvalidShape, offset, "tensor.shape",
          "non-tiled layouts must not pad logical extents"));
    }
  }
  return {};
}

std::expected<void, FormatError> validate_quantizer_layout(
    TensorRecord const& tensor, std::uint64_t offset) {
  switch (tensor.quantizer) {
    case LogicalQuantizerId::Q4G64V0:
      if (tensor.storage != StorageClass::Int4Grouped) {
        return std::unexpected(make_error(
            FormatErrorCode::InvalidStorageQuantizerPair, offset,
            "tensor.storage", "Q4G64 requires int4 grouped storage"));
      }
      if (tensor.layout != PhysicalLayoutId::CudaQ4G64V0) {
        return std::unexpected(make_error(
            FormatErrorCode::InvalidQuantizerLayoutPair, offset, "tensor.layout",
            "Q4G64 requires cuda_q4g64_v0"));
      }
      return {};
    case LogicalQuantizerId::Q8G32V0:
      if (tensor.storage != StorageClass::Int8Grouped) {
        return std::unexpected(make_error(
            FormatErrorCode::InvalidStorageQuantizerPair, offset,
            "tensor.storage", "Q8G32 requires int8 grouped storage"));
      }
      if (tensor.layout != PhysicalLayoutId::CudaQ8G32V0) {
        return std::unexpected(make_error(
            FormatErrorCode::InvalidQuantizerLayoutPair, offset, "tensor.layout",
            "Q8G32 requires cuda_q8g32_v0"));
      }
      return {};
    case LogicalQuantizerId::None:
      if (tensor.storage == StorageClass::Fp32) {
        if (tensor.layout != PhysicalLayoutId::CudaFp32VectorV0) {
          return std::unexpected(make_error(
              FormatErrorCode::InvalidQuantizerLayoutPair, offset,
              "tensor.layout",
              "FP32 generated tensors require cuda_fp32_vector_v0"));
        }
        return {};
      }
      if (tensor.storage != StorageClass::Bf16) {
        return std::unexpected(make_error(
            FormatErrorCode::InvalidStorageQuantizerPair, offset,
            "tensor.storage", "unquantized tensors require BF16 or FP32 storage"));
      }
      if (tensor.layout != PhysicalLayoutId::CudaBf16DenseTileV0 &&
          tensor.layout != PhysicalLayoutId::CudaBf16RowMajorV0 &&
          tensor.layout != PhysicalLayoutId::CudaBf16VectorV0 &&
          tensor.layout != PhysicalLayoutId::CudaBf16TapMajorV0) {
        return std::unexpected(make_error(
            FormatErrorCode::InvalidQuantizerLayoutPair, offset, "tensor.layout",
            "unquantized tensors require a BF16 weight layout"));
      }
      return {};
  }
  return std::unexpected(make_error(FormatErrorCode::UnknownEnum, offset,
                                    "tensor.quantizer",
                                    "unsupported quantizer"));
}

std::expected<void, FormatError> validate_tensor(TensorRecord const& tensor,
                                                 std::uint64_t offset) {
  if (tensor.logical_name.empty()) {
    return std::unexpected(make_error(FormatErrorCode::InvalidShape, offset,
                                      "tensor.logical_name",
                                      "logical identity must be nonempty"));
  }
  if (auto st = validate_shape(tensor.shape, offset, "tensor.shape"); !st) {
    return st;
  }
  if (!layout_is_weight(tensor.layout)) {
    return std::unexpected(make_error(
        FormatErrorCode::InvalidQuantizerLayoutPair, offset, "tensor.layout",
        "weight tensors cannot use a state layout"));
  }
  if (auto st = validate_quantizer_layout(tensor, offset); !st) {
    return st;
  }
  if (auto st = validate_mapping_for_tensor(tensor, offset); !st) {
    return st;
  }
  auto const payload_n = expected_payload_bytes(tensor, offset);
  if (!payload_n) {
    return std::unexpected(payload_n.error());
  }
  auto const scale_n = expected_scale_bytes(tensor, offset);
  if (!scale_n) {
    return std::unexpected(scale_n.error());
  }
  if (auto st = require_span_alignment(tensor.payload.offset,
                                       tensor.payload.length,
                                       "tensor.payload");
      !st) {
    return st;
  }
  if (auto st = require_span_alignment(tensor.scales.offset, tensor.scales.length,
                                       "tensor.scales");
      !st) {
    return st;
  }
  if (tensor.payload.length != *payload_n) {
    return std::unexpected(make_error(FormatErrorCode::InvalidSpan, offset,
                                      "tensor.payload.length",
                                      "payload length does not match layout"));
  }
  if (tensor.scales.length != *scale_n) {
    return std::unexpected(make_error(FormatErrorCode::InvalidSpan, offset,
                                      "tensor.scales.length",
                                      "scale length does not match quantizer"));
  }
  if (*payload_n != 0) {
    if (auto st = checked_add(tensor.payload.offset, tensor.payload.length,
                              offset, "tensor.payload");
        !st) {
      return std::unexpected(st.error());
    }
  }
  if (*scale_n != 0) {
    if (auto st = checked_add(tensor.scales.offset, tensor.scales.length, offset,
                              "tensor.scales");
        !st) {
      return std::unexpected(st.error());
    }
  }
  return {};
}

std::expected<void, FormatError> validate_precision(
    PrecisionPolicyRecord const& policy, std::uint64_t offset) {
  if (policy.id != PrecisionPolicyId::V0) {
    return std::unexpected(make_error(FormatErrorCode::InvalidPrecisionPolicy,
                                      offset, "precision.id",
                                      "only precision policy V0 is defined"));
  }
  struct Required {
    PrecisionDomain domain;
    ArithmeticDtype dtype;
    char const* field;
  };
  constexpr Required kRequired[] = {
      {PrecisionDomain::ResidualStream, ArithmeticDtype::Fp32,
       "precision.residual"},
      {PrecisionDomain::NormalizedActivation, ArithmeticDtype::Bf16,
       "precision.normalized"},
      {PrecisionDomain::ProjectionStaging, ArithmeticDtype::Bf16,
       "precision.projection_staging"},
      {PrecisionDomain::DotProductAccumulator, ArithmeticDtype::Fp32,
       "precision.accumulator"},
      {PrecisionDomain::NonlinearIntermediate, ArithmeticDtype::Fp32,
       "precision.nonlinear"},
      {PrecisionDomain::GdnRecurrentState, ArithmeticDtype::Fp32,
       "precision.gdn_s"},
      {PrecisionDomain::ConvolutionHistory, ArithmeticDtype::Bf16,
       "precision.conv"},
      {PrecisionDomain::KvCache, ArithmeticDtype::Bf16, "precision.kv"},
      {PrecisionDomain::Logits, ArithmeticDtype::Fp32, "precision.logits"},
      {PrecisionDomain::WeightScale, ArithmeticDtype::Fp16,
       "precision.weight_scale"},
  };
  if (policy.bindings.size() != std::size(kRequired)) {
    return std::unexpected(make_error(
        FormatErrorCode::InvalidPrecisionPolicy, offset, "precision.bindings",
        "V0 policy must list every precision domain once"));
  }
  for (auto const& req : kRequired) {
    int seen = 0;
    for (auto const& b : policy.bindings) {
      if (b.domain == req.domain) {
        ++seen;
        if (b.dtype != req.dtype) {
          return std::unexpected(make_error(
              FormatErrorCode::InvalidPrecisionPolicy, offset, req.field,
              "V0 dtype mismatch for precision domain"));
        }
      }
    }
    if (seen != 1) {
      return std::unexpected(make_error(FormatErrorCode::InvalidPrecisionPolicy,
                                        offset, req.field,
                                        "precision domain missing or duplicated"));
    }
  }
  return {};
}

std::expected<void, FormatError> validate_state(StateAllocation const& s,
                                                std::uint64_t offset) {
  if (s.live_payload_present) {
    return std::unexpected(make_error(
        FormatErrorCode::LiveStateForbidden, offset, "state.live_payload",
        "artifact describes allocation schema, never live state"));
  }
  if (auto st = validate_shape(s.shape_per_layer, offset, "state.shape"); !st) {
    return st;
  }
  if (s.layer_count == 0 || s.component_count == 0) {
    return std::unexpected(make_error(FormatErrorCode::InvalidStateAllocation,
                                      offset, "state.layer_count",
                                      "layer and component counts must be > 0"));
  }
  auto const elem = element_size(s.dtype);
  if (elem == 0) {
    return std::unexpected(make_error(FormatErrorCode::UnknownEnum, offset,
                                      "state.dtype", "unknown arithmetic dtype"));
  }
  auto const dims =
      s.shape_per_layer.logical_dims(offset, "state.shape");
  if (!dims) {
    return std::unexpected(dims.error());
  }
  auto const elems = checked_product(*dims, offset, "state.shape");
  if (!elems) {
    return std::unexpected(elems.error());
  }
  for (std::uint8_t i = 0; i < s.shape_per_layer.rank; ++i) {
    if (s.shape_per_layer.padded[i] != s.shape_per_layer.logical[i]) {
      return std::unexpected(make_error(FormatErrorCode::InvalidShape, offset,
                                        "state.shape",
                                        "state schema does not pad extents"));
    }
  }
  switch (s.kind) {
    case StateKind::GdnS: {
      if (s.dtype != ArithmeticDtype::Fp32 ||
          s.layout != PhysicalLayoutId::CudaFp32GdnSHvKV0 ||
          s.shape_per_layer.rank != 3 || s.shape_per_layer.logical[0] != 48 ||
          s.shape_per_layer.logical[1] != 128 ||
          s.shape_per_layer.logical[2] != 128 || s.component_count != 1 ||
          s.declared_capacity != 0 || s.bytes_per_token != 0 ||
          s.populated_length_distinct_from_capacity || s.layer_count != 48) {
        return std::unexpected(make_error(
            FormatErrorCode::InvalidStateAllocation, offset, "state.gdn_s",
            "GDN S must be FP32 [head,value,key]=[48,128,128] x 48 layers"));
      }
      break;
    }
    case StateKind::ConvolutionHistory: {
      if (s.dtype != ArithmeticDtype::Bf16 ||
          s.layout != PhysicalLayoutId::CudaBf16ConvHistoryV0 ||
          s.shape_per_layer.rank != 2 || s.shape_per_layer.logical[0] != 3 ||
          s.shape_per_layer.logical[1] != 10240 || s.component_count != 1 ||
          s.declared_capacity != 0 || s.bytes_per_token != 0 ||
          s.populated_length_distinct_from_capacity || s.layer_count != 48) {
        return std::unexpected(make_error(
            FormatErrorCode::InvalidStateAllocation, offset, "state.conv",
            "convolution history must be BF16 [3,10240] x 48 layers"));
      }
      break;
    }
    case StateKind::KvCache: {
      if (s.dtype != ArithmeticDtype::Bf16 ||
          s.layout != PhysicalLayoutId::CudaBf16KvCacheV0 ||
          s.shape_per_layer.rank != 2 || s.shape_per_layer.logical[0] != 4 ||
          s.shape_per_layer.logical[1] != 256 || s.component_count != 2 ||
          !s.populated_length_distinct_from_capacity || s.layer_count != 16) {
        return std::unexpected(make_error(
            FormatErrorCode::InvalidStateAllocation, offset, "state.kv",
            "KV must be BF16 [kv_head,head_dim]=[4,256], 2 components, 16 layers"));
      }
      break;
    }
  }
  auto const per_layer_elems =
      checked_mul(*elems, s.component_count, offset, "state.component_count");
  if (!per_layer_elems) {
    return std::unexpected(per_layer_elems.error());
  }
  if (s.kind == StateKind::KvCache) {
    auto const per_token_layer =
        checked_mul(*per_layer_elems, elem, offset, "state.bytes_per_token");
    if (!per_token_layer) {
      return std::unexpected(per_token_layer.error());
    }
    auto const coeff =
        checked_mul(*per_token_layer, s.layer_count, offset, "state.bytes_per_token");
    if (!coeff) {
      return std::unexpected(coeff.error());
    }
    if (s.bytes_per_token != *coeff) {
      return std::unexpected(make_error(
          FormatErrorCode::InvalidStateAllocation, offset, "state.bytes_per_token",
          "KV bytes-per-token coefficient mismatch"));
    }
    if (s.declared_capacity == 0) {
      if (s.bytes_per_layer != 0 || s.total_bytes != 0) {
        return std::unexpected(make_error(
            FormatErrorCode::InvalidStateAllocation, offset, "state.total_bytes",
            "KV without declared capacity must leave totals unset"));
      }
    } else {
      auto const layer_bytes = checked_mul(*per_token_layer, s.declared_capacity,
                                           offset, "state.bytes_per_layer");
      if (!layer_bytes) {
        return std::unexpected(layer_bytes.error());
      }
      auto const total =
          checked_mul(*coeff, s.declared_capacity, offset, "state.total_bytes");
      if (!total) {
        return std::unexpected(total.error());
      }
      if (s.bytes_per_layer != *layer_bytes || s.total_bytes != *total) {
        return std::unexpected(make_error(
            FormatErrorCode::InvalidStateAllocation, offset, "state.total_bytes",
            "KV allocation bytes do not match capacity"));
      }
    }
    return {};
  }
  auto const layer_bytes =
      checked_mul(*per_layer_elems, elem, offset, "state.bytes_per_layer");
  if (!layer_bytes) {
    return std::unexpected(layer_bytes.error());
  }
  auto const total =
      checked_mul(*layer_bytes, s.layer_count, offset, "state.total_bytes");
  if (!total) {
    return std::unexpected(total.error());
  }
  if (s.bytes_per_layer != *layer_bytes || s.total_bytes != *total) {
    return std::unexpected(make_error(FormatErrorCode::InvalidStateAllocation,
                                      offset, "state.total_bytes",
                                      "state byte counts do not match geometry"));
  }
  return {};
}

std::expected<void, FormatError> validate_scratch(ScratchAllocation const& s,
                                                  std::uint64_t offset) {
  if (s.bytes == 0) {
    return std::unexpected(make_error(FormatErrorCode::InvalidScratchAllocation,
                                      offset, "scratch.bytes",
                                      "scratch allocation must be nonzero"));
  }
  auto const elem = element_size(s.dtype);
  if (elem == 0) {
    return std::unexpected(make_error(FormatErrorCode::UnknownEnum, offset,
                                      "scratch.dtype", "unknown dtype"));
  }
  if ((s.bytes % elem) != 0) {
    return std::unexpected(make_error(
        FormatErrorCode::InvalidScratchAllocation, offset, "scratch.bytes",
        "scratch byte count must be aligned to its arithmetic dtype"));
  }
  ArithmeticDtype expected_dtype{};
  std::uint64_t expected_bytes = 0;
  switch (s.kind) {
    case ScratchKind::ResidualH:
    case ScratchKind::ResidualHMid:
      expected_dtype = ArithmeticDtype::Fp32;
      expected_bytes = 20480;
      break;
    case ScratchKind::NormalizedHidden:
      expected_dtype = ArithmeticDtype::Bf16;
      expected_bytes = 10240;
      break;
    case ScratchKind::GdnWorkspace:
      expected_dtype = ArithmeticDtype::Fp32;
      expected_bytes = 107264;
      break;
    case ScratchKind::AttentionWorkspace:
      expected_dtype = ArithmeticDtype::Fp32;
      expected_bytes = 78016;
      break;
    case ScratchKind::MlpSwiglu:
      expected_dtype = ArithmeticDtype::Bf16;
      expected_bytes = 34816;
      break;
    case ScratchKind::Logits:
      expected_dtype = ArithmeticDtype::Fp32;
      expected_bytes = 993280;
      break;
  }
  if (s.dtype != expected_dtype || s.bytes != expected_bytes) {
    return std::unexpected(make_error(
        FormatErrorCode::InvalidScratchAllocation, offset, "scratch",
        "scratch record does not match the V0 language schema"));
  }
  return {};
}

}  // namespace

PrecisionPolicyRecord v0_precision_policy() {
  return PrecisionPolicyRecord{
      .id = PrecisionPolicyId::V0,
      .bindings =
          {
              {PrecisionDomain::ResidualStream, ArithmeticDtype::Fp32},
              {PrecisionDomain::NormalizedActivation, ArithmeticDtype::Bf16},
              {PrecisionDomain::ProjectionStaging, ArithmeticDtype::Bf16},
              {PrecisionDomain::DotProductAccumulator, ArithmeticDtype::Fp32},
              {PrecisionDomain::NonlinearIntermediate, ArithmeticDtype::Fp32},
              {PrecisionDomain::GdnRecurrentState, ArithmeticDtype::Fp32},
              {PrecisionDomain::ConvolutionHistory, ArithmeticDtype::Bf16},
              {PrecisionDomain::KvCache, ArithmeticDtype::Bf16},
              {PrecisionDomain::Logits, ArithmeticDtype::Fp32},
              {PrecisionDomain::WeightScale, ArithmeticDtype::Fp16},
          },
  };
}

std::array<StateAllocation, 3> v0_language_state_schema() {
  StateAllocation gdn{};
  gdn.kind = StateKind::GdnS;
  gdn.dtype = ArithmeticDtype::Fp32;
  gdn.layout = PhysicalLayoutId::CudaFp32GdnSHvKV0;
  gdn.shape_per_layer.rank = 3;
  gdn.shape_per_layer.logical[0] = 48;
  gdn.shape_per_layer.logical[1] = 128;
  gdn.shape_per_layer.logical[2] = 128;
  gdn.shape_per_layer.padded = gdn.shape_per_layer.logical;
  gdn.layer_count = 48;
  gdn.component_count = 1;
  gdn.bytes_per_layer = 3145728;
  gdn.total_bytes = 150994944;

  StateAllocation conv{};
  conv.kind = StateKind::ConvolutionHistory;
  conv.dtype = ArithmeticDtype::Bf16;
  conv.layout = PhysicalLayoutId::CudaBf16ConvHistoryV0;
  conv.shape_per_layer.rank = 2;
  conv.shape_per_layer.logical[0] = 3;
  conv.shape_per_layer.logical[1] = 10240;
  conv.shape_per_layer.padded = conv.shape_per_layer.logical;
  conv.layer_count = 48;
  conv.component_count = 1;
  conv.bytes_per_layer = 61440;
  conv.total_bytes = 2949120;

  StateAllocation kv{};
  kv.kind = StateKind::KvCache;
  kv.dtype = ArithmeticDtype::Bf16;
  kv.layout = PhysicalLayoutId::CudaBf16KvCacheV0;
  kv.shape_per_layer.rank = 2;
  kv.shape_per_layer.logical[0] = 4;
  kv.shape_per_layer.logical[1] = 256;
  kv.shape_per_layer.padded = kv.shape_per_layer.logical;
  kv.layer_count = 16;
  kv.component_count = 2;
  kv.bytes_per_token = 65536;
  kv.populated_length_distinct_from_capacity = true;

  return {gdn, conv, kv};
}

std::array<ScratchAllocation, 7> v0_language_scratch_schema() {
  return {
      ScratchAllocation{.kind = ScratchKind::ResidualH,
                        .dtype = ArithmeticDtype::Fp32,
                        .bytes = 20480},
      ScratchAllocation{.kind = ScratchKind::ResidualHMid,
                        .dtype = ArithmeticDtype::Fp32,
                        .bytes = 20480},
      ScratchAllocation{.kind = ScratchKind::NormalizedHidden,
                        .dtype = ArithmeticDtype::Bf16,
                        .bytes = 10240},
      ScratchAllocation{.kind = ScratchKind::GdnWorkspace,
                        .dtype = ArithmeticDtype::Fp32,
                        .bytes = 107264},
      ScratchAllocation{.kind = ScratchKind::AttentionWorkspace,
                        .dtype = ArithmeticDtype::Fp32,
                        .bytes = 78016},
      ScratchAllocation{.kind = ScratchKind::MlpSwiglu,
                        .dtype = ArithmeticDtype::Bf16,
                        .bytes = 34816},
      ScratchAllocation{.kind = ScratchKind::Logits,
                        .dtype = ArithmeticDtype::Fp32,
                        .bytes = 993280},
  };
}

std::expected<std::uint64_t, FormatError> expected_payload_bytes(
    TensorRecord const& tensor, std::uint64_t offset) {
  if (auto st = validate_shape(tensor.shape, offset, "tensor.shape"); !st) {
    return std::unexpected(st.error());
  }
  if (layout_is_tiled_dense(tensor.layout)) {
    if (tensor.shape.rank != 2) {
      return std::unexpected(make_error(FormatErrorCode::InvalidShape, offset,
                                        "tensor.shape", "tiled payload needs [N,K]"));
    }
    auto const tiles_n =
        tensor.shape.padded[0] / static_cast<std::uint64_t>(kDenseTileRows);
    auto const tiles_k =
        tensor.shape.padded[1] / static_cast<std::uint64_t>(kDenseTileK);
    auto const tiles = checked_mul(tiles_n, tiles_k, offset, "tensor.tiles");
    if (!tiles) {
      return std::unexpected(tiles.error());
    }
    auto const rows =
        checked_mul(*tiles, kDenseTileRows, offset, "tensor.tile_rows");
    if (!rows) {
      return std::unexpected(rows.error());
    }
    std::uint64_t packed = 0;
    if (tensor.layout == PhysicalLayoutId::CudaQ4G64V0) {
      packed = kQ4PackedBytesPerTileRow;
    } else if (tensor.layout == PhysicalLayoutId::CudaQ8G32V0) {
      packed = kQ8PackedBytesPerTileRow;
    } else {
      packed = kBf16PackedBytesPerTileRow;
    }
    return checked_mul(*rows, packed, offset, "tensor.payload");
  }
  auto const dims = tensor.shape.logical_dims(offset, "tensor.shape");
  if (!dims) {
    return std::unexpected(dims.error());
  }
  auto const elems = checked_product(*dims, offset, "tensor.shape");
  if (!elems) {
    return std::unexpected(elems.error());
  }
  auto const elem_size = tensor.storage == StorageClass::Fp32 ? kFp32Size
                                                              : kBf16Size;
  return checked_mul(*elems, elem_size, offset, "tensor.payload");
}

std::expected<std::uint64_t, FormatError> expected_scale_bytes(
    TensorRecord const& tensor, std::uint64_t offset) {
  if (auto st = validate_shape(tensor.shape, offset, "tensor.shape"); !st) {
    return std::unexpected(st.error());
  }
  if (tensor.quantizer == LogicalQuantizerId::None) {
    return 0;
  }
  if (tensor.shape.rank != 2) {
    return std::unexpected(make_error(FormatErrorCode::InvalidShape, offset,
                                      "tensor.shape",
                                      "quantized tensors are rank-2"));
  }
  std::uint64_t group = 0;
  if (tensor.quantizer == LogicalQuantizerId::Q4G64V0) {
    group = kQ4GroupSize;
  } else if (tensor.quantizer == LogicalQuantizerId::Q8G32V0) {
    group = kQ8GroupSize;
  } else {
    return std::unexpected(make_error(FormatErrorCode::UnknownEnum, offset,
                                      "tensor.quantizer",
                                      "unsupported quantizer"));
  }
  if (group == 0 || tensor.shape.padded[1] % group != 0) {
    return std::unexpected(make_error(FormatErrorCode::InvalidShape, offset,
                                      "tensor.shape.k",
                                      "padded K must divide the quantizer group"));
  }
  auto const groups_k = tensor.shape.padded[1] / group;
  auto const count =
      checked_mul(tensor.shape.padded[0], groups_k, offset, "tensor.scales");
  if (!count) {
    return std::unexpected(count.error());
  }
  return checked_mul(*count, kFp16Size, offset, "tensor.scales");
}

std::expected<void, FormatError> validate_shared_binding_ownership(
    ArtifactSchema const& schema, std::uint64_t offset) {
  std::unordered_map<std::uint32_t, std::uint32_t> owner_by_alias;
  std::unordered_set<std::uint32_t> aliases;
  std::unordered_set<std::uint32_t> owners;
  owner_by_alias.reserve(schema.shared_bindings.size());
  aliases.reserve(schema.shared_bindings.size());
  owners.reserve(schema.shared_bindings.size());

  for (auto const& binding : schema.shared_bindings) {
    if (binding.owner_tensor_id == binding.alias_tensor_id) {
      return std::unexpected(make_error(FormatErrorCode::SharedBinding, offset,
                                        "shared.alias",
                                        "owner and alias must differ"));
    }
    if (find_tensor(schema, binding.owner_tensor_id) == nullptr ||
        find_tensor(schema, binding.alias_tensor_id) == nullptr) {
      return std::unexpected(make_error(
          FormatErrorCode::SharedBinding, offset, "shared.owner",
          "shared binding references a missing tensor"));
    }
    auto const [it, inserted] = owner_by_alias.emplace(
        binding.alias_tensor_id, binding.owner_tensor_id);
    if (!inserted) {
      return std::unexpected(make_error(
          FormatErrorCode::SharedBinding, offset, "shared.alias",
          it->second == binding.owner_tensor_id
              ? "duplicate shared binding for alias"
              : "alias has multiple owners"));
    }
    aliases.insert(binding.alias_tensor_id);
    owners.insert(binding.owner_tensor_id);
  }

  for (auto owner : owners) {
    if (aliases.contains(owner)) {
      return std::unexpected(make_error(
          FormatErrorCode::SharedBinding, offset, "shared.owner",
          "an alias cannot own another alias; chains and cycles are forbidden"));
    }
  }
  return {};
}

std::expected<std::uint32_t, FormatError> canonical_owner_tensor_id(
    ArtifactSchema const& schema, std::uint32_t tensor_id,
    std::uint64_t offset) {
  if (find_tensor(schema, tensor_id) == nullptr) {
    return std::unexpected(make_error(FormatErrorCode::SharedBinding, offset,
                                      "shared.tensor",
                                      "tensor is missing"));
  }
  if (auto st = validate_shared_binding_ownership(schema, offset); !st) {
    return std::unexpected(st.error());
  }
  for (auto const& binding : schema.shared_bindings) {
    if (binding.alias_tensor_id == tensor_id) {
      return binding.owner_tensor_id;
    }
  }
  return tensor_id;
}

std::expected<std::size_t, FormatError> encoded_size(
    ContainerHeader const&) {
  return static_cast<std::size_t>(kHeaderSizeV0);
}

std::expected<void, FormatError> encode(ContainerHeader const& header,
                                        std::span<std::byte> out) {
  if (out.size() < kHeaderSizeV0) {
    return std::unexpected(make_error(FormatErrorCode::BufferTooSmall, 0,
                                      "header", "header requires 64 bytes"));
  }
  ByteWriter w{out.first(kHeaderSizeV0)};
  std::array<std::byte, 8> magic{};
  for (std::size_t i = 0; i < header.magic.size(); ++i) {
    magic[i] = static_cast<std::byte>(header.magic[i]);
  }
  if (auto st = w.bytes(magic, "header.magic"); !st) {
    return st;
  }
  if (auto st = w.u16(header.container_version, "header.container_version");
      !st) {
    return st;
  }
  if (auto st = w.u16(header.manifest_version, "header.manifest_version"); !st) {
    return st;
  }
  if (auto st = w.u16(header.header_bytes, "header.header_bytes"); !st) {
    return st;
  }
  if (auto st = w.u16(0, "header.reserved0"); !st) {
    return st;
  }
  if (auto st = w.u64(header.manifest_offset, "header.manifest_offset"); !st) {
    return st;
  }
  if (auto st = w.u64(header.manifest_length, "header.manifest_length"); !st) {
    return st;
  }
  return w.zeros(32, "header.reserved");
}

std::expected<ContainerHeader, FormatError> decode_header(
    std::span<std::byte const> in) {
  if (in.size() < kHeaderSizeV0) {
    return std::unexpected(make_error(FormatErrorCode::Truncated, 0, "header",
                                      "header requires 64 bytes"));
  }
  ByteReader r{in.first(kHeaderSizeV0)};
  ContainerHeader header{};
  std::array<std::byte, 8> magic{};
  if (auto st = r.bytes(magic, "header.magic"); !st) {
    return std::unexpected(st.error());
  }
  for (std::size_t i = 0; i < header.magic.size(); ++i) {
    header.magic[i] = static_cast<std::uint8_t>(magic[i]);
  }
  auto const container = r.u16("header.container_version");
  if (!container) {
    return std::unexpected(container.error());
  }
  auto const manifest = r.u16("header.manifest_version");
  if (!manifest) {
    return std::unexpected(manifest.error());
  }
  auto const header_bytes = r.u16("header.header_bytes");
  if (!header_bytes) {
    return std::unexpected(header_bytes.error());
  }
  if (auto st = r.expect_zeros(2, "header.reserved0"); !st) {
    return std::unexpected(st.error());
  }
  auto const moff = r.u64("header.manifest_offset");
  if (!moff) {
    return std::unexpected(moff.error());
  }
  auto const mlen = r.u64("header.manifest_length");
  if (!mlen) {
    return std::unexpected(mlen.error());
  }
  if (auto st = r.expect_zeros(32, "header.reserved"); !st) {
    return std::unexpected(st.error());
  }
  header.container_version = *container;
  header.manifest_version = *manifest;
  header.header_bytes = *header_bytes;
  header.manifest_offset = *moff;
  header.manifest_length = *mlen;
  if (auto st = validate_header(header, 0); !st) {
    return std::unexpected(st.error());
  }
  return header;
}

std::expected<void, FormatError> validate_header(ContainerHeader const& header,
                                                 std::uint64_t offset) {
  if (header.magic != kMagic) {
    return std::unexpected(make_error(FormatErrorCode::BadMagic, offset,
                                      "header.magic",
                                      "magic is not QW38FMT"));
  }
  if (header.container_version != kContainerVersionV0) {
    return std::unexpected(make_error(
        FormatErrorCode::UnsupportedContainerVersion, offset,
        "header.container_version", "only container version 1 is supported"));
  }
  if (header.manifest_version != kManifestVersionV0) {
    return std::unexpected(make_error(
        FormatErrorCode::UnsupportedManifestVersion, offset,
        "header.manifest_version", "only manifest version 1 is supported"));
  }
  if (header.header_bytes != kHeaderSizeV0) {
    return std::unexpected(make_error(FormatErrorCode::InvalidHeader, offset,
                                      "header.header_bytes",
                                      "V0 header is 64 bytes"));
  }
  if (header.manifest_offset < kHeaderSizeV0) {
    return std::unexpected(make_error(FormatErrorCode::InvalidHeader, offset,
                                      "header.manifest_offset",
                                      "manifest must start after the header"));
  }
  if (header.manifest_length > kMaxManifestBytesV0) {
    return std::unexpected(make_error(
        FormatErrorCode::ResourceLimitExceeded, offset,
        "header.manifest_length", "manifest exceeds the V0 size limit"));
  }
  if (auto st = checked_add(header.manifest_offset, header.manifest_length,
                            offset, "header.manifest");
      !st) {
    return std::unexpected(st.error());
  }
  return {};
}

std::expected<std::size_t, FormatError> encoded_size(
    ArtifactSchema const& schema) {
  SizeAcc acc;
  if (auto st = acc.add(2, 0, "manifest.version"); !st) {
    return std::unexpected(st.error());
  }
  auto const ident_n =
      checked_add(2, schema.compiler.ident.size(), 0, "compiler.ident");
  if (!ident_n) {
    return std::unexpected(ident_n.error());
  }
  if (auto st = acc.add(*ident_n + 16, 0, "compiler"); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = acc.add(32 * 3, 0, "hashes"); !st) {
    return std::unexpected(st.error());
  }
  auto const bind_bytes = checked_mul(schema.precision.bindings.size(), 4, 0,
                                      "precision.bindings");
  if (!bind_bytes) {
    return std::unexpected(bind_bytes.error());
  }
  if (auto st = acc.add(4 + *bind_bytes, 0, "precision"); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = acc.add(2, 0, "scope"); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = acc.add(4, 0, "tensor_count"); !st) {
    return std::unexpected(st.error());
  }
  for (auto const& t : schema.tensors) {
    auto const n = tensor_encoded_bytes(t, 0);
    if (!n) {
      return std::unexpected(n.error());
    }
    if (auto st = acc.add(*n, 0, "tensors"); !st) {
      return std::unexpected(st.error());
    }
  }
  if (auto st = acc.add(4, 0, "shared_count"); !st) {
    return std::unexpected(st.error());
  }
  auto const shared_n =
      checked_mul(schema.shared_bindings.size(), 12, 0, "shared_bindings");
  if (!shared_n) {
    return std::unexpected(shared_n.error());
  }
  if (auto st = acc.add(*shared_n, 0, "shared_bindings"); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = acc.add(4, 0, "graph_count"); !st) {
    return std::unexpected(st.error());
  }
  auto const graph_n =
      checked_mul(schema.graph_bindings.size(), 16, 0, "graph_bindings");
  if (!graph_n) {
    return std::unexpected(graph_n.error());
  }
  if (auto st = acc.add(*graph_n, 0, "graph_bindings"); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = acc.add(4, 0, "state_count"); !st) {
    return std::unexpected(st.error());
  }
  for (auto const& s : schema.state) {
    auto const n = state_encoded_bytes(s, 0);
    if (!n) {
      return std::unexpected(n.error());
    }
    if (auto st = acc.add(*n, 0, "state"); !st) {
      return std::unexpected(st.error());
    }
  }
  if (auto st = acc.add(4, 0, "scratch_count"); !st) {
    return std::unexpected(st.error());
  }
  auto const scratch_n =
      checked_mul(schema.scratch.size(), 16, 0, "scratch");
  if (!scratch_n) {
    return std::unexpected(scratch_n.error());
  }
  if (auto st = acc.add(*scratch_n, 0, "scratch"); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = acc.add(4, 0, "integrity_count"); !st) {
    return std::unexpected(st.error());
  }
  auto const integ_n =
      checked_mul(schema.integrity.size(), kIntegrityRecordBytes, 0,
                  "integrity");
  if (!integ_n) {
    return std::unexpected(integ_n.error());
  }
  if (auto st = acc.add(*integ_n, 0, "integrity"); !st) {
    return std::unexpected(st.error());
  }
  auto const total = acc.finish(0, "schema");
  if (!total) {
    return std::unexpected(total.error());
  }
  if (*total > kMaxManifestBytesV0) {
    return std::unexpected(make_error(
        FormatErrorCode::ResourceLimitExceeded, 0, "schema",
        "encoded manifest exceeds the V0 size limit"));
  }
  return total;
}

std::expected<void, FormatError> encode(ArtifactSchema const& schema,
                                        std::span<std::byte> out) {
  auto const n = encoded_size(schema);
  if (!n) {
    return std::unexpected(n.error());
  }
  if (out.size() < *n) {
    return std::unexpected(make_error(FormatErrorCode::BufferTooSmall, 0,
                                      "schema", "encode buffer too small"));
  }
  ByteWriter w{out.first(*n)};
  if (auto st = w.u16(schema.manifest_version, "manifest.version"); !st) {
    return st;
  }
  if (auto st = write_compiler(w, schema.compiler); !st) {
    return st;
  }
  if (auto st = write_hash(w, schema.source_hash, "source_hash"); !st) {
    return st;
  }
  if (auto st = write_hash(w, schema.config_hash, "config_hash"); !st) {
    return st;
  }
  if (auto st = write_hash(w, schema.tokenizer_hash, "tokenizer_hash"); !st) {
    return st;
  }
  if (auto st = write_precision(w, schema.precision); !st) {
    return st;
  }
  if (auto st = w.u16(std::to_underlying(schema.scope), "scope"); !st) {
    return st;
  }
  if (schema.tensors.size() > std::numeric_limits<std::uint32_t>::max()) {
    return std::unexpected(make_error(FormatErrorCode::Overflow, w.offset(),
                                      "tensors", "too many tensors"));
  }
  if (auto st = w.u32(static_cast<std::uint32_t>(schema.tensors.size()),
                      "tensor_count");
      !st) {
    return st;
  }
  for (auto const& t : schema.tensors) {
    if (auto st = write_tensor(w, t); !st) {
      return st;
    }
  }
  if (schema.shared_bindings.size() > std::numeric_limits<std::uint32_t>::max()) {
    return std::unexpected(make_error(FormatErrorCode::Overflow, w.offset(),
                                      "shared_bindings",
                                      "too many shared bindings"));
  }
  if (auto st = w.u32(static_cast<std::uint32_t>(schema.shared_bindings.size()),
                      "shared_count");
      !st) {
    return st;
  }
  for (auto const& b : schema.shared_bindings) {
    if (auto st = write_shared(w, b); !st) {
      return st;
    }
  }
  if (schema.graph_bindings.size() > std::numeric_limits<std::uint32_t>::max()) {
    return std::unexpected(make_error(FormatErrorCode::Overflow, w.offset(),
                                      "graph_bindings",
                                      "too many graph bindings"));
  }
  if (auto st = w.u32(static_cast<std::uint32_t>(schema.graph_bindings.size()),
                      "graph_count");
      !st) {
    return st;
  }
  for (auto const& b : schema.graph_bindings) {
    if (auto st = write_graph(w, b); !st) {
      return st;
    }
  }
  if (schema.state.size() > std::numeric_limits<std::uint32_t>::max()) {
    return std::unexpected(make_error(FormatErrorCode::Overflow, w.offset(),
                                      "state", "too many state records"));
  }
  if (auto st =
          w.u32(static_cast<std::uint32_t>(schema.state.size()), "state_count");
      !st) {
    return st;
  }
  for (auto const& s : schema.state) {
    if (auto st = write_state(w, s); !st) {
      return st;
    }
  }
  if (schema.scratch.size() > std::numeric_limits<std::uint32_t>::max()) {
    return std::unexpected(make_error(FormatErrorCode::Overflow, w.offset(),
                                      "scratch", "too many scratch records"));
  }
  if (auto st = w.u32(static_cast<std::uint32_t>(schema.scratch.size()),
                      "scratch_count");
      !st) {
    return st;
  }
  for (auto const& s : schema.scratch) {
    if (auto st = write_scratch(w, s); !st) {
      return st;
    }
  }
  if (schema.integrity.size() > std::numeric_limits<std::uint32_t>::max()) {
    return std::unexpected(make_error(FormatErrorCode::Overflow, w.offset(),
                                      "integrity", "too many integrity records"));
  }
  if (auto st = w.u32(static_cast<std::uint32_t>(schema.integrity.size()),
                      "integrity_count");
      !st) {
    return st;
  }
  for (auto const& rec : schema.integrity) {
    if (auto st = write_integrity(w, rec); !st) {
      return st;
    }
  }
  if (w.offset() != *n) {
    return std::unexpected(make_error(
        FormatErrorCode::Overflow, w.offset(), "schema",
        "encoded_size does not match bytes written"));
  }
  return {};
}

std::expected<ArtifactSchema, FormatError> decode_schema(
    std::span<std::byte const> in) try {
  if (in.size() > kMaxManifestBytesV0) {
    return std::unexpected(make_error(
        FormatErrorCode::ResourceLimitExceeded, 0, "manifest",
        "manifest exceeds the V0 size limit"));
  }
  ByteReader r{in};
  ArtifactSchema schema{};
  auto const version = r.u16("manifest.version");
  if (!version) {
    return std::unexpected(version.error());
  }
  if (*version != kManifestVersionV0) {
    return std::unexpected(make_error(
        FormatErrorCode::UnsupportedManifestVersion, 0, "manifest.version",
        "only manifest version 1 is supported"));
  }
  schema.manifest_version = *version;
  auto const compiler = read_compiler(r);
  if (!compiler) {
    return std::unexpected(compiler.error());
  }
  schema.compiler = *compiler;
  auto const source = read_hash(r, "source_hash");
  if (!source) {
    return std::unexpected(source.error());
  }
  schema.source_hash = *source;
  auto const config = read_hash(r, "config_hash");
  if (!config) {
    return std::unexpected(config.error());
  }
  schema.config_hash = *config;
  auto const tokenizer = read_hash(r, "tokenizer_hash");
  if (!tokenizer) {
    return std::unexpected(tokenizer.error());
  }
  schema.tokenizer_hash = *tokenizer;
  auto const precision = read_precision(r);
  if (!precision) {
    return std::unexpected(precision.error());
  }
  schema.precision = *precision;
  auto const scope = r.u16("scope");
  if (!scope) {
    return std::unexpected(scope.error());
  }
  auto decoded_scope = decode_semantic_scope(*scope, r.offset() - 2, "scope");
  if (!decoded_scope) {
    return std::unexpected(decoded_scope.error());
  }
  schema.scope = *decoded_scope;

  auto const n_tensors = r.u32("tensor_count");
  if (!n_tensors) {
    return std::unexpected(n_tensors.error());
  }
  if (auto st = preflight_count(r, *n_tensors, 90, kMaxTensorRecordsV0,
                                "tensor_count");
      !st) {
    return std::unexpected(st.error());
  }
  schema.tensors.reserve(*n_tensors);
  for (std::uint32_t i = 0; i < *n_tensors; ++i) {
    auto t = read_tensor(r);
    if (!t) {
      return std::unexpected(t.error());
    }
    schema.tensors.push_back(std::move(*t));
  }
  auto const n_shared = r.u32("shared_count");
  if (!n_shared) {
    return std::unexpected(n_shared.error());
  }
  if (auto st = preflight_count(r, *n_shared, 12, kMaxSharedBindingsV0,
                                "shared_count");
      !st) {
    return std::unexpected(st.error());
  }
  schema.shared_bindings.reserve(*n_shared);
  for (std::uint32_t i = 0; i < *n_shared; ++i) {
    auto b = read_shared(r);
    if (!b) {
      return std::unexpected(b.error());
    }
    schema.shared_bindings.push_back(*b);
  }
  auto const n_graph = r.u32("graph_count");
  if (!n_graph) {
    return std::unexpected(n_graph.error());
  }
  if (auto st = preflight_count(r, *n_graph, 16, kMaxGraphBindingsV0,
                                "graph_count");
      !st) {
    return std::unexpected(st.error());
  }
  schema.graph_bindings.reserve(*n_graph);
  for (std::uint32_t i = 0; i < *n_graph; ++i) {
    auto b = read_graph(r);
    if (!b) {
      return std::unexpected(b.error());
    }
    schema.graph_bindings.push_back(*b);
  }
  auto const n_state = r.u32("state_count");
  if (!n_state) {
    return std::unexpected(n_state.error());
  }
  if (auto st = preflight_count(r, *n_state, 72, kMaxStateAllocationsV0,
                                "state_count");
      !st) {
    return std::unexpected(st.error());
  }
  schema.state.reserve(*n_state);
  for (std::uint32_t i = 0; i < *n_state; ++i) {
    auto s = read_state(r);
    if (!s) {
      return std::unexpected(s.error());
    }
    schema.state.push_back(*s);
  }
  auto const n_scratch = r.u32("scratch_count");
  if (!n_scratch) {
    return std::unexpected(n_scratch.error());
  }
  if (auto st = preflight_count(r, *n_scratch, 16, kMaxScratchAllocationsV0,
                                "scratch_count");
      !st) {
    return std::unexpected(st.error());
  }
  schema.scratch.reserve(*n_scratch);
  for (std::uint32_t i = 0; i < *n_scratch; ++i) {
    auto s = read_scratch(r);
    if (!s) {
      return std::unexpected(s.error());
    }
    schema.scratch.push_back(*s);
  }
  auto const n_int = r.u32("integrity_count");
  if (!n_int) {
    return std::unexpected(n_int.error());
  }
  if (auto st = preflight_count(r, *n_int, kIntegrityRecordBytes,
                                kMaxIntegrityRecordsV0, "integrity_count");
      !st) {
    return std::unexpected(st.error());
  }
  schema.integrity.reserve(*n_int);
  for (std::uint32_t i = 0; i < *n_int; ++i) {
    auto rec = read_integrity(r);
    if (!rec) {
      return std::unexpected(rec.error());
    }
    schema.integrity.push_back(*rec);
  }
  if (auto st = r.expect_consumed(); !st) {
    return std::unexpected(st.error());
  }
  if (auto st = validate_schema(schema, 0); !st) {
    return std::unexpected(st.error());
  }
  return schema;
} catch (std::bad_alloc const&) {
  return std::unexpected(
      FormatError{.code = FormatErrorCode::AllocationFailure});
} catch (std::length_error const&) {
  return std::unexpected(make_error(
      FormatErrorCode::ResourceLimitExceeded, 0, "manifest",
      "container size is not representable while decoding schema"));
}

std::expected<void, FormatError> validate_schema(ArtifactSchema const& schema,
                                                 std::uint64_t offset) {
  if (schema.manifest_version != kManifestVersionV0) {
    return std::unexpected(make_error(
        FormatErrorCode::UnsupportedManifestVersion, offset, "manifest.version",
        "only manifest version 1 is supported"));
  }
  if (schema.precision.bindings.size() > kMaxPrecisionBindingsV0 ||
      schema.tensors.size() > kMaxTensorRecordsV0 ||
      schema.shared_bindings.size() > kMaxSharedBindingsV0 ||
      schema.graph_bindings.size() > kMaxGraphBindingsV0 ||
      schema.state.size() > kMaxStateAllocationsV0 ||
      schema.scratch.size() > kMaxScratchAllocationsV0 ||
      schema.integrity.size() > kMaxIntegrityRecordsV0) {
    return std::unexpected(make_error(
        FormatErrorCode::ResourceLimitExceeded, offset, "schema",
        "one or more record counts exceed V0 limits"));
  }
  if (schema.compiler.ident.empty()) {
    return std::unexpected(make_error(FormatErrorCode::InvalidHeader, offset,
                                      "compiler.ident",
                                      "compiler revision ident is required"));
  }
  if (!is_known(schema.scope)) {
    return std::unexpected(make_error(FormatErrorCode::UnknownEnum, offset,
                                      "scope", "unsupported semantic scope"));
  }
  if (auto st = validate_precision(schema.precision, offset); !st) {
    return st;
  }
  for (std::size_t i = 0; i < schema.tensors.size(); ++i) {
    if (auto st = validate_tensor(schema.tensors[i], offset); !st) {
      return st;
    }
    for (std::size_t j = i + 1; j < schema.tensors.size(); ++j) {
      if (schema.tensors[i].tensor_id == schema.tensors[j].tensor_id) {
        return std::unexpected(make_error(FormatErrorCode::DuplicateId, offset,
                                          "tensor.id",
                                          "tensor ids must be unique"));
      }
      if (schema.tensors[i].logical_name == schema.tensors[j].logical_name) {
        return std::unexpected(make_error(FormatErrorCode::DuplicateId, offset,
                                          "tensor.logical_name",
                                          "logical identities must be unique"));
      }
    }
  }
  if (auto st = validate_shared_binding_ownership(schema, offset); !st) {
    return st;
  }
  for (auto const& g : schema.graph_bindings) {
    if (find_tensor(schema, g.tensor_id) == nullptr) {
      return std::unexpected(make_error(FormatErrorCode::InvalidGraphBinding,
                                        offset, "graph.tensor",
                                        "graph binding references missing tensor"));
    }
    if ((g.kind == SemanticNodeKind::GatedAttention ||
         g.kind == SemanticNodeKind::GatedDeltaNet ||
         g.kind == SemanticNodeKind::Mlp) &&
        g.layer_index == kNoLayerIndex) {
      return std::unexpected(make_error(FormatErrorCode::InvalidGraphBinding,
                                        offset, "graph.layer",
                                        "layer-scoped nodes need a layer index"));
    }
    if (g.kind == SemanticNodeKind::MtpMix &&
        schema.scope != SemanticScope::LanguagePlusMtpDescriptors) {
      return std::unexpected(make_error(
          FormatErrorCode::InvalidGraphBinding, offset, "graph.kind",
          "MTP nodes require language-plus-MTP descriptor scope"));
    }
    if (g.kind == SemanticNodeKind::Embed &&
        g.role != TensorRole::EmbeddingTable) {
      return std::unexpected(make_error(FormatErrorCode::InvalidGraphBinding,
                                        offset, "graph.role",
                                        "EMBED binds an embedding table"));
    }
    if (g.kind == SemanticNodeKind::LmHead &&
        g.role != TensorRole::LmHeadWeight &&
        g.role != TensorRole::FinalLanguageNorm &&
        g.role != TensorRole::MtpNorm) {
      return std::unexpected(make_error(FormatErrorCode::InvalidGraphBinding,
                                        offset, "graph.role",
                                        "LM_HEAD binds its vocabulary matrix and norm"));
    }
    bool const valid_norm_node =
        (g.role != TensorRole::AdditiveNorm ||
         g.kind == SemanticNodeKind::GatedAttention ||
         g.kind == SemanticNodeKind::GatedDeltaNet ||
         g.kind == SemanticNodeKind::Mlp ||
         g.kind == SemanticNodeKind::MtpMix) &&
        (g.role != TensorRole::QkNorm ||
         g.kind == SemanticNodeKind::GatedAttention) &&
        (g.role != TensorRole::GdnGatedNorm ||
         g.kind == SemanticNodeKind::GatedDeltaNet) &&
        (g.role != TensorRole::FinalLanguageNorm ||
         g.kind == SemanticNodeKind::LmHead) &&
        (g.role != TensorRole::MtpNorm ||
         g.kind == SemanticNodeKind::LmHead);
    if (!valid_norm_node) {
      return std::unexpected(make_error(FormatErrorCode::InvalidGraphBinding,
                                        offset, "graph.role",
                                        "norm role is invalid for graph node"));
    }
  }
  bool seen_gdn = false;
  bool seen_conv = false;
  bool seen_kv = false;
  for (auto const& s : schema.state) {
    if (auto st = validate_state(s, offset); !st) {
      return st;
    }
    if (s.kind == StateKind::GdnS) {
      if (seen_gdn) {
        return std::unexpected(make_error(FormatErrorCode::DuplicateId, offset,
                                          "state.kind",
                                          "duplicate GDN S allocation"));
      }
      seen_gdn = true;
    } else if (s.kind == StateKind::ConvolutionHistory) {
      if (seen_conv) {
        return std::unexpected(make_error(FormatErrorCode::DuplicateId, offset,
                                          "state.kind",
                                          "duplicate convolution history"));
      }
      seen_conv = true;
    } else if (s.kind == StateKind::KvCache) {
      if (seen_kv) {
        return std::unexpected(make_error(FormatErrorCode::DuplicateId, offset,
                                          "state.kind", "duplicate KV cache"));
      }
      seen_kv = true;
    }
  }
  if (!seen_gdn || !seen_conv || !seen_kv || schema.state.size() != 3) {
    return std::unexpected(make_error(
        FormatErrorCode::InvalidStateAllocation, offset, "state",
        "V0 language state requires exactly GDN S, convolution history, and KV"));
  }
  for (std::size_t i = 0; i < schema.scratch.size(); ++i) {
    if (auto st = validate_scratch(schema.scratch[i], offset); !st) {
      return st;
    }
    for (std::size_t j = i + 1; j < schema.scratch.size(); ++j) {
      if (schema.scratch[i].kind == schema.scratch[j].kind) {
        return std::unexpected(make_error(FormatErrorCode::DuplicateId, offset,
                                          "scratch.kind",
                                          "duplicate scratch kind"));
      }
    }
  }
  if (schema.scratch.size() != v0_language_scratch_schema().size()) {
    return std::unexpected(make_error(
        FormatErrorCode::InvalidScratchAllocation, offset, "scratch",
        "V0 language scratch requires every declared scratch kind exactly once"));
  }
  for (auto const& b : schema.shared_bindings) {
    auto const* owner = find_tensor(schema, b.owner_tensor_id);
    auto const* alias = find_tensor(schema, b.alias_tensor_id);
    if (!tensors_share_payload(*owner, *alias)) {
      return std::unexpected(make_error(
          FormatErrorCode::SharedBinding, offset, "shared.payload",
          "alias must reuse the owner's shape, layout, and spans"));
    }
    bool const owner_embed =
        has_graph_role(schema, owner->tensor_id, SemanticNodeKind::Embed,
                       TensorRole::EmbeddingTable);
    bool const owner_head =
        has_graph_role(schema, owner->tensor_id, SemanticNodeKind::LmHead,
                       TensorRole::LmHeadWeight);
    bool const alias_embed =
        has_graph_role(schema, alias->tensor_id, SemanticNodeKind::Embed,
                       TensorRole::EmbeddingTable);
    bool const alias_head =
        has_graph_role(schema, alias->tensor_id, SemanticNodeKind::LmHead,
                       TensorRole::LmHeadWeight);
    if ((owner_embed && (alias_head || owner_head)) ||
        (owner_head && (alias_embed || owner_embed)) ||
        (alias_embed && alias_head)) {
      return std::unexpected(make_error(
          FormatErrorCode::SharedBinding, offset, "shared.role",
          "embeddings and lm_head remain untied"));
    }
    if (b.role == SharedBindingRole::MtpEmbeddingAlias) {
      if (schema.scope != SemanticScope::LanguagePlusMtpDescriptors) {
        return std::unexpected(make_error(
            FormatErrorCode::SharedBinding, offset, "shared.role",
            "MTP embedding alias requires MTP descriptor scope"));
      }
      if (!owner_embed) {
        return std::unexpected(make_error(
            FormatErrorCode::SharedBinding, offset, "shared.owner",
            "MTP embedding alias must bind to the embedding table"));
      }
    }
    if (b.role == SharedBindingRole::MtpLmHeadAlias) {
      if (schema.scope != SemanticScope::LanguagePlusMtpDescriptors) {
        return std::unexpected(make_error(
            FormatErrorCode::SharedBinding, offset, "shared.role",
            "MTP head alias requires MTP descriptor scope"));
      }
      if (!owner_head) {
        return std::unexpected(make_error(
            FormatErrorCode::SharedBinding, offset, "shared.owner",
            "MTP head alias must bind to lm_head"));
      }
    }
  }
  for (std::size_t i = 0; i < schema.tensors.size(); ++i) {
    for (std::size_t j = i + 1; j < schema.tensors.size(); ++j) {
      auto const& a = schema.tensors[i];
      auto const& b = schema.tensors[j];
      if (a.payload.empty() || b.payload.empty()) {
        continue;
      }
      if (a.payload == b.payload) {
        bool bound = false;
        for (auto const& sb : schema.shared_bindings) {
          if ((sb.owner_tensor_id == a.tensor_id &&
               sb.alias_tensor_id == b.tensor_id) ||
              (sb.owner_tensor_id == b.tensor_id &&
               sb.alias_tensor_id == a.tensor_id)) {
            bound = true;
            break;
          }
        }
        if (!bound) {
          return std::unexpected(make_error(
              FormatErrorCode::SharedBinding, offset, "tensor.payload",
              "identical payload spans require a shared binding"));
        }
      }
    }
  }
  for (auto const& rec : schema.integrity) {
    if (auto st = checked_add(rec.region.offset, rec.region.length, offset,
                              "integrity.region");
        !st) {
      return std::unexpected(st.error());
    }
    if (rec.kind == IntegrityKind::Sha256PayloadSpan ||
        rec.kind == IntegrityKind::Sha256ScaleSpan) {
      if (find_tensor(schema, rec.tensor_id) == nullptr) {
        return std::unexpected(make_error(
            FormatErrorCode::InvalidSpan, offset, "integrity.tensor",
            "payload integrity record needs a tensor id"));
      }
    }
  }
  return {};
}

}  // namespace qw38::format
