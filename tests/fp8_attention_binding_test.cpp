#include "format/reader.hpp"
#include "format/writer.hpp"
#include "runtime/prefill.hpp"
#include "runtime/runtime.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <vector>
#include <unistd.h>

namespace {
void require(bool ok, char const* message) {
  if (!ok) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); }
}
template<class T, class E> T take(std::expected<T, E> result) {
  if (!result) { std::cerr << error_message(result.error()) << '\n'; std::exit(1); }
  return std::move(*result);
}
bool same(qw38::runtime::SessionSnapshot const& a,
          qw38::runtime::SessionSnapshot const& b) {
  return a.gdn_s == b.gdn_s && a.conv_history == b.conv_history &&
      a.kv == b.kv && a.conv_cursor == b.conv_cursor &&
      a.gdn_position == b.gdn_position && a.kv_populated == b.kv_populated &&
      a.token_position == b.token_position;
}
}

int main() {
  using namespace qw38;
  auto filename = std::getenv("QW38_AUTHORITY_ARTIFACT");
  require(filename, "FP8 authority artifact required");
  auto source = take(format::Artifact::open(filename));
  auto rt = take(runtime::Runtime::create());
  auto path = std::filesystem::temp_directory_path() /
      ("qw38-fp8-attention-binding-" + std::to_string(getpid()) + ".qw38");
  // Keep real layer-3 weights, graph roles, layouts and Session arenas. Replace
  // only one projection group with valid zero BF16 weights for mixed fixtures.
  for (unsigned variant = 0; variant < 3; ++variant) {
    auto schema = source.schema();
    std::erase_if(schema.graph_bindings, [](auto const& b) {
      return b.layer_index != 3 ||
          (b.kind != format::SemanticNodeKind::GatedAttention &&
           b.kind != format::SemanticNodeKind::Mlp);
    });
    std::erase_if(schema.tensors, [&](auto const& t) {
      return !std::ranges::any_of(schema.graph_bindings,
          [&](auto const& b) { return b.tensor_id == t.tensor_id; });
    });
    schema.shared_bindings.clear();
    schema.integrity.clear();
    schema.scope = format::SemanticScope::PrimaryLanguage;
    auto replace = [&](auto const& t) {
      return variant == 1 ? t.logical_name == runtime::attn_o_name(3) :
          variant == 2 && (t.logical_name == runtime::attn_q_name(3) ||
              t.logical_name == runtime::attn_k_name(3) ||
              t.logical_name == runtime::attn_v_name(3));
    };
    for (auto& t : schema.tensors) {
      t.payload = {}; t.scales = {};
      if (!replace(t)) continue;
      t.storage = format::StorageClass::Bf16;
      t.quantizer = format::LogicalQuantizerId::None;
      t.layout = format::PhysicalLayoutId::CudaBf16DenseTileV0;
      t.mapping = {.kind = format::MappingKind::DenseTileNK,
          .tile_rows = 8, .tile_k = 256, .group_size = 0,
          .packed_bytes_per_tile_row = 512};
    }
    auto writer = take(format::ArtifactWriter::create(path, schema));
    for (auto const& t : schema.tensors) {
      if (replace(t)) {
        std::vector<std::byte> zeros(take(format::expected_payload_bytes(t)));
        require(bool(writer.write_span(t.logical_name, format::SpanKind::Payload, zeros)), "BF16 fixture write");
      } else {
        require(bool(writer.write_span(t.logical_name, format::SpanKind::Payload,
            take(source.payload(t.logical_name)))), "real weight fixture write");
        auto scales = take(source.scales(t.logical_name));
        if (!scales.empty()) require(bool(writer.write_span(t.logical_name,
            format::SpanKind::Scales, scales)), "scale fixture write");
      }
    }
    take(writer.finalize());
    auto model = take(rt.load(path));
    std::filesystem::remove(path);
    auto session = take(rt.create_session(model, 3));
    auto before = take(session.save());
    auto plan = runtime::bind_prefill_attention_layer(model, session, 3, rt.stream());
    if (variant == 0) require(bool(plan), "uniform FP8 positive binding control");
    else require(!plan && plan.error().field == "prefill.attention.layout",
        "mixed FP8 input/output rejected by policy boundary");
    require(same(before, take(session.save())), "binding leaves device state and metadata unchanged");
    std::cout << "attention binding variant=" << variant << " PASS\n";
  }
}
