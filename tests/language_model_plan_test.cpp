#include "runtime/language_model.hpp"

#include <cstdint>
#include <iostream>

int main() {
  using namespace qw38;
  format::ArtifactSchema schema;
  schema.scope = format::SemanticScope::LanguagePlusMtpDescriptors;
  for (std::uint32_t id = 0; id < 130; ++id) {
    auto const layer = id == 0 || id == 129
        ? format::kNoLayerIndex : (id - 1) / 2;
    auto const kind = id == 0 ? format::SemanticNodeKind::Embed
        : id == 129 ? format::SemanticNodeKind::LmHead
        : id % 2 == 0 ? format::SemanticNodeKind::Mlp
        : runtime::is_gdn_language_layer(layer)
            ? format::SemanticNodeKind::GatedDeltaNet
            : format::SemanticNodeKind::GatedAttention;
    schema.graph_bindings.push_back({.instance_id = id,
                                     .kind = kind,
                                     .layer_index = layer,
                                     .tensor_id = id + 1});
  }
  schema.graph_bindings.push_back({.instance_id = 132,
                                   .kind = format::SemanticNodeKind::GatedAttention,
                                   .layer_index = 0,
                                   .tensor_id = 200});
  auto check = [](bool ok, char const* what) {
    if (!ok) std::cerr << "FAIL: " << what << '\n';
    return ok;
  };
  bool ok = check(static_cast<bool>(runtime::validate_primary_language_graph(schema)),
                  "130 language instances with retained MTP descriptors");
  auto changed = schema;
  changed.graph_bindings[7].kind = format::SemanticNodeKind::GatedDeltaNet;
  ok &= check(!runtime::validate_primary_language_graph(changed),
              "wrong mixer family rejected");
  changed = schema;
  changed.graph_bindings[128].layer_index = format::kNoLayerIndex;
  ok &= check(!runtime::validate_primary_language_graph(changed),
              "wrong final MLP layer rejected");
  changed = schema;
  changed.graph_bindings.erase(changed.graph_bindings.begin() + 85);
  ok &= check(!runtime::validate_primary_language_graph(changed),
              "missing language instance rejected");
  changed = schema;
  changed.scope = format::SemanticScope::PrimaryLanguage;
  ok &= check(!runtime::validate_primary_language_graph(changed),
              "disabled MTP descriptor scope required");
  return ok ? 0 : 1;
}
