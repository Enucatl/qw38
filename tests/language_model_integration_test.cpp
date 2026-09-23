#include "runtime/language_model.hpp"
#include "runtime/runtime.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace qw38::runtime {
struct LanguageModelPlanTestAccess {
  static void* head_output(LanguageModelPlan const& plan) {
    return plan.head_.output.pointer;
  }
  static void set_head_output(LanguageModelPlan& plan, void* pointer) {
    plan.head_.output.pointer = pointer;
  }
};
}  // namespace qw38::runtime

namespace {

bool equal_state(qw38::runtime::SessionSnapshot const& a,
                 qw38::runtime::SessionSnapshot const& b) {
  return a.gdn_s == b.gdn_s && a.conv_history == b.conv_history &&
         a.kv == b.kv && a.conv_cursor == b.conv_cursor &&
         a.gdn_position == b.gdn_position && a.kv_populated == b.kv_populated &&
         a.token_position == b.token_position;
}

bool write_logits(std::uint32_t position, std::span<float const> logits) {
  auto const* directory = std::getenv("QW38_ENGINE_LOGITS_DIR");
  if (directory == nullptr) return true;
  auto path = std::filesystem::path(directory) /
              ("engine-logits-" + std::to_string(position) + ".f32");
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<char const*>(logits.data()),
               static_cast<std::streamsize>(logits.size_bytes()));
  return static_cast<bool>(output);
}

}  // namespace

int main() {
  auto const* artifact = std::getenv("QW38_AUTHORITY_ARTIFACT");
  if (artifact == nullptr) {
    std::cout << "authority decode skipped (set QW38_AUTHORITY_ARTIFACT)\n";
    return 0;
  }
  using namespace qw38::runtime;
  auto fail = [](char const* what) {
    std::cerr << "FAIL: " << what << '\n';
    return 1;
  };
  auto runtime = Runtime::create();
  if (!runtime) return fail("create runtime");
  auto model = runtime->load(artifact);
  if (!model) {
    std::cerr << error_message(model.error()) << '\n';
    return fail("load authority");
  }
  auto session = runtime->create_session(*model, 2);
  if (!session) return fail("create session");
  auto plan = LanguageModelPlan::bind(*model, *session, runtime->stream());
  if (!plan) {
    std::cerr << error_message(plan.error()) << '\n';
    return fail("bind full language model");
  }
  auto bad_token = plan->decode_token(kVocab, 0);
  auto bad_position = plan->decode_token(1, 1);
  if (bad_token || bad_position) return fail("token and position errors");
  auto first = plan->decode_token(1, 0);
  if (!first || first->logits.size() != kVocab ||
      !std::ranges::all_of(first->logits, [](float x) { return std::isfinite(x); }) ||
      first->argmax >= kVocab) {
    if (!first) std::cerr << error_message(first.error()) << '\n';
    return fail("first token FP32 logits");
  }
  if (!write_logits(0, first->logits)) return fail("write first logits");
  auto initial = session->save();
  if (!initial || initial->token_position != 1 ||
      !std::ranges::all_of(initial->gdn_position,
                           [](auto x) { return x == 1; }) ||
      !std::ranges::all_of(initial->kv_populated,
                           [](auto x) { return x == 1; })) {
    return fail("complete token state boundary");
  }
  auto const saved_head = LanguageModelPlanTestAccess::head_output(*plan);
  LanguageModelPlanTestAccess::set_head_output(*plan, nullptr);
  auto injected = plan->decode_token(2, 1);
  LanguageModelPlanTestAccess::set_head_output(*plan, saved_head);
  if (injected || session->save() || plan->decode_token(2, 1)) {
    return fail("late failure must poison execution and snapshots");
  }
  if (!session->restore(*initial)) return fail("restore after late failure");
  auto continued = plan->decode_token(2, 1);
  if (!continued) {
    std::cerr << error_message(continued.error()) << '\n';
    return fail("continuation after restore");
  }
  if (!write_logits(1, continued->logits)) return fail("write continued logits");
  std::vector<float> continued_logits(continued->logits.begin(),
                                      continued->logits.end());
  auto continued_state = session->save();
  if (!continued_state || continued_state->token_position != 2) {
    return fail("continued complete token state");
  }
  if (!session->reset()) return fail("reset");
  auto reset_state = session->save();
  if (!reset_state || reset_state->token_position != 0) {
    return fail("reset token position");
  }
  std::array<std::uint32_t, 2> const prompt{1, 2};
  auto replay = plan->setup_prompt_slow(prompt);
  if (!replay || !std::equal(replay->logits.begin(), replay->logits.end(),
                             continued_logits.begin(), continued_logits.end())) {
    return fail("deterministic repeated decode logits");
  }
  auto replay_state = session->save();
  if (!replay_state || !equal_state(*replay_state, *continued_state)) {
    return fail("reset replay persistent bytes and metadata");
  }
  std::cout << "authority primary-language decode, restore and replay passed\n";
  return 0;
}
