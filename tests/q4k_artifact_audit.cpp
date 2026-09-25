#include "format/reader.hpp"

#include <algorithm>
#include <iostream>

int main(int argc, char** argv) {
  using namespace qw38::format;
  if (argc != 3) return 2;
  auto old = Artifact::open(argv[1]);
  auto next = Artifact::open(argv[2]);
  if (!old || !next) return 1;
  auto const& a = old->schema();
  auto const& b = next->schema();
  if (a.precision.id != PrecisionPolicyId::CandidateV1 ||
      b.precision.id != PrecisionPolicyId::CandidateV2 ||
      a.precision.bindings != b.precision.bindings || a.state != b.state ||
      a.scratch != b.scratch || a.graph_bindings != b.graph_bindings ||
      a.shared_bindings != b.shared_bindings || a.source_hash != b.source_hash ||
      a.config_hash != b.config_hash || a.tokenizer_hash != b.tokenizer_hash ||
      a.tensors.size() != b.tensors.size()) return 1;
  std::size_t changed = 0;
  std::uint64_t extra = 0;
  for (auto const& t : a.tensors) {
    auto const* u = next->find_tensor(t.logical_name);
    if (!u || t.shape != u->shape || t.storage != u->storage) return 1;
    if (u->quantizer == LogicalQuantizerId::Q4KCandidateV2) {
      if (t.quantizer != LogicalQuantizerId::Q4G64CandidateV1 ||
          !t.logical_name.starts_with("model.language_model.layers.") ||
          (t.logical_name.find(".mlp.gate_proj.weight") == std::string::npos &&
           t.logical_name.find(".mlp.up_proj.weight") == std::string::npos &&
           t.logical_name.find(".mlp.down_proj.weight") == std::string::npos)) return 1;
      ++changed;
      extra += u->payload.length + u->scales.length - t.payload.length - t.scales.length;
      continue;
    }
    if (t.quantizer != u->quantizer || t.layout != u->layout || t.mapping != u->mapping) return 1;
    auto p = old->payload(t.logical_name);
    auto q = next->payload(t.logical_name);
    if (!p || !q || !std::ranges::equal(*p, *q)) return 1;
    if (t.scales.length != u->scales.length) return 1;
    if (t.scales.length) {
      auto s = old->scales(t.logical_name);
      auto v = next->scales(t.logical_name);
      if (!s || !v || !std::ranges::equal(*s, *v)) return 1;
    }
  }
  if (changed != 192 || extra != 510ULL * 1024 * 1024) return 1;
  std::cout << "{\"changed_mlp_matrices\":" << changed
            << ",\"extra_weight_bytes\":" << extra
            << ",\"other_payloads_and_metadata\":\"byte_identical\","
               "\"precision_state_scratch_graph\":\"unchanged\"}\n";
}
