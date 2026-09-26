#include "runtime/language_model.hpp"
#include "runtime/runtime.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qw38::runtime {
struct LanguageModelPlanTestAccess {
  static void* head_output(LanguageModelPlan const& plan) {
    return plan.head_.output.pointer;
  }
  static void set_head_output(LanguageModelPlan& plan, void* pointer) {
    plan.head_.output.pointer = pointer;
  }
  static void const* prefill_head_codes(LanguageModelPlan const& plan) {
    return plan.prefill_->head.codes;
  }
  static void set_prefill_head_codes(LanguageModelPlan& plan, void const* pointer) {
    plan.prefill_->head.codes = pointer;
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
  auto session = runtime->create_session(*model, 3);
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
  if (!session->reset()) return fail("reset before prefill");
  std::array<std::uint64_t, 2> const rows{0, 1};
  std::uint32_t delivered = 0;
  auto prefilled = plan->prefill_tokens(prompt, rows,
      [&](std::uint64_t row, std::span<float const> logits)
          -> std::expected<void, Error> {
        if (row != delivered || logits.size() != kVocab ||
            !std::ranges::all_of(logits, [](float x) { return std::isfinite(x); }))
          return std::unexpected(make_error(ErrorCode::Internal, "prefill.test",
                                            "requested row mismatch"));
        ++delivered;
        return {};
      });
  if (!prefilled || delivered != 2) {
    if (!prefilled) std::cerr << error_message(prefilled.error()) << '\n';
    return fail("full-model prefill requested rows");
  }
  std::vector<float> prefill_logits(prefilled->logits.begin(), prefilled->logits.end());
  auto prefill_state = session->save();
  if (!prefill_state || prefill_state->token_position != 2 ||
      !std::ranges::all_of(prefill_state->gdn_position,
                           [](auto x) { return x == 2; }) ||
      !std::ranges::all_of(prefill_state->kv_populated,
                           [](auto x) { return x == 2; }))
    return fail("prefill complete token state");
  if (!session->reset()) return fail("prefill replay reset");
  auto prefill_replay = plan->prefill_tokens(prompt);
  auto replay_prefill_state = session->save();
  if (!prefill_replay || !replay_prefill_state ||
      !std::equal(prefill_replay->logits.begin(), prefill_replay->logits.end(),
                  prefill_logits.begin()) ||
      !equal_state(*replay_prefill_state, *prefill_state))
    return fail("same-schedule prefill replay");
  if (!session->reset()) return fail("prefill failure reset");
  auto const* saved_codes = LanguageModelPlanTestAccess::prefill_head_codes(*plan);
  LanguageModelPlanTestAccess::set_prefill_head_codes(*plan, nullptr);
  auto failed_prefill = plan->prefill_tokens(prompt);
  LanguageModelPlanTestAccess::set_prefill_head_codes(*plan, saved_codes);
  if (failed_prefill || session->save() || plan->prefill_tokens(prompt))
    return fail("late prefill failure must poison execution and snapshots");
  if (!session->restore(*prefill_state)) return fail("restore after prefill failure");
  if (!plan->decode_token(3, 2)) return fail("prefill to decode handoff");
  if (!session->reset()) return fail("mixed schedule reset");
  if (!plan->decode_token(1, 0)) return fail("mixed schedule prefix");
  std::array<std::uint32_t, 1> const tail{2};
  if (!plan->prefill_tokens(tail) || !plan->decode_token(3, 2))
    return fail("nonempty-session prefill to decode handoff");

  auto boundary_session = runtime->create_session(*model, 259);
  if (!boundary_session) return fail("create boundary session");
  auto boundary_plan = LanguageModelPlan::bind(*model, *boundary_session,
                                               runtime->stream());
  if (!boundary_plan) return fail("bind boundary plan");
  std::vector<std::uint32_t> sequence(258);
  for (std::size_t i = 0; i < sequence.size(); ++i)
    sequence[i] = static_cast<std::uint32_t>(1 + i % 3);
  auto before = boundary_session->save();
  if (!before) return fail("save boundary start");
  std::array<std::uint64_t, 2> const duplicate_rows{0, 0};
  std::array<std::uint64_t, 1> const outside_rows{258};
  auto sink = [](std::uint64_t, std::span<float const>)
      -> std::expected<void, Error> { return {}; };
  auto duplicate = boundary_plan->prefill_tokens(sequence, duplicate_rows, sink);
  auto outside = boundary_plan->prefill_tokens(sequence, outside_rows, sink);
  auto no_sink = boundary_plan->prefill_tokens(sequence, outside_rows);
  auto after_invalid = boundary_session->save();
  if (duplicate || outside || no_sink || !after_invalid ||
      !equal_state(*before, *after_invalid))
    return fail("invalid requested rows must not mutate session");
  std::array<std::uint64_t, 4> const checkpoints{0, 255, 256, 257};
  std::array<std::vector<float>, 4> selected;
  std::size_t next_row = 0;
  auto selected_run = boundary_plan->prefill_tokens(sequence, checkpoints,
      [&](std::uint64_t position, std::span<float const> logits)
          -> std::expected<void, Error> {
        if (next_row >= checkpoints.size() || position != checkpoints[next_row] ||
            logits.size() != kVocab)
          return std::unexpected(make_error(ErrorCode::Internal, "prefill.test",
                                            "wrong requested row"));
        selected[next_row++].assign(logits.begin(), logits.end());
        return {};
      });
  if (!selected_run || next_row != checkpoints.size() ||
      std::equal(selected[0].begin(), selected[0].end(), selected[1].begin()))
    return fail("requested row contents across chunk boundary");
  if (std::equal(selected[1].begin(), selected[1].end(), selected[2].begin()) ||
      std::equal(selected[2].begin(), selected[2].end(), selected[3].begin()))
    return fail("requested rows on opposite sides of chunk boundary differ");
  for (std::size_t i : {1u, 3u}) {
    if (!boundary_session->reset()) return fail("reset selected-row reference");
    auto prefix = std::span<std::uint32_t const>(sequence.data(), checkpoints[i] + 1);
    auto reference = boundary_plan->prefill_tokens(prefix);
    if (!reference || !std::equal(reference->logits.begin(), reference->logits.end(),
                                  selected[i].begin()))
      return fail("selected row differs from same-schedule final row");
  }
  if (!boundary_session->reset() ||
      !boundary_plan->decode_token(sequence[0], 0))
    return fail("nonempty selected-row setup");
  std::array<std::uint64_t, 2> const nonzero_rows{256, 257};
  std::array<std::vector<float>, 2> nonzero_selected;
  std::size_t nonzero_index = 0;
  auto nonzero_run = boundary_plan->prefill_tokens(
      std::span<std::uint32_t const>(sequence.data() + 1, 257), nonzero_rows,
      [&](std::uint64_t position, std::span<float const> logits)
          -> std::expected<void, Error> {
        if (nonzero_index >= nonzero_rows.size() ||
            position != nonzero_rows[nonzero_index])
          return std::unexpected(make_error(ErrorCode::Internal, "prefill.test",
                                            "wrong absolute row"));
        nonzero_selected[nonzero_index++].assign(logits.begin(), logits.end());
        return {};
      });
  if (!nonzero_run || nonzero_index != nonzero_rows.size())
    return fail("nonempty selected rows");
  if (std::equal(nonzero_selected[0].begin(), nonzero_selected[0].end(),
                 nonzero_selected[1].begin()))
    return fail("nonempty selected rows differ");
  for (std::size_t i : {1u}) {
    if (!boundary_session->reset() ||
        !boundary_plan->decode_token(sequence[0], 0))
      return fail("reset nonempty selected-row reference");
    auto prefix = std::span<std::uint32_t const>(sequence.data() + 1,
                                                 nonzero_rows[i]);
    auto reference = boundary_plan->prefill_tokens(prefix);
    if (!reference || !std::equal(reference->logits.begin(), reference->logits.end(),
                                  nonzero_selected[i].begin()))
      return fail("nonempty selected row differs from same-schedule final row");
  }
  if (!boundary_session->reset()) return fail("reset before sink failures");
  auto clean = boundary_session->save();
  if (!clean) return fail("save before sink failures");
  auto two = std::span<std::uint32_t const>(sequence.data(), 2);
  std::array<std::uint64_t, 1> const first_row{0};
  auto returned_failure = boundary_plan->prefill_tokens(two, first_row,
      [](std::uint64_t, std::span<float const>) -> std::expected<void, Error> {
        return std::unexpected(make_error(ErrorCode::Internal, "prefill.test",
                                          "injected sink failure"));
      });
  if (returned_failure || boundary_session->save() ||
      boundary_plan->prefill_tokens(two))
    return fail("returned sink failure must poison session");
  if (!boundary_session->restore(*clean)) return fail("restore after sink error");
  auto thrown_failure = boundary_plan->prefill_tokens(two, first_row,
      [](std::uint64_t, std::span<float const>) -> std::expected<void, Error> {
        throw std::runtime_error("injected sink exception");
      });
  if (thrown_failure || boundary_session->save() ||
      boundary_plan->prefill_tokens(two))
    return fail("thrown sink failure must poison session");
  if (!boundary_session->restore(*clean) || !boundary_plan->prefill_tokens(two))
    return fail("restore after sink exception");

  auto control_session = runtime->create_session(*model, 3);
  auto moving_session = runtime->create_session(*model, 3);
  if (!control_session || !moving_session) return fail("create movement sessions");
  auto control_plan = LanguageModelPlan::bind(*model, *control_session,
                                              runtime->stream());
  auto moving_plan = LanguageModelPlan::bind(*model, *moving_session,
                                             runtime->stream());
  if (!control_plan || !moving_plan) return fail("bind movement plans");
  auto control_result = control_plan->prefill_tokens(prompt);
  auto control_state = control_session->save();
  Session moved = std::move(*moving_session);
  auto moved_result = moving_plan->prefill_tokens(prompt);
  auto moved_state = moved.save();
  if (!control_result || !control_state || !moved_result || !moved_state ||
      !std::equal(control_result->logits.begin(), control_result->logits.end(),
                  moved_result->logits.begin()) ||
      !equal_state(*control_state, *moved_state))
    return fail("prefill after session move before initialization");
  Session moved_again = std::move(moved);
  if (!moved_again.reset()) return fail("reset session after second move");
  auto moved_again_result = moving_plan->prefill_tokens(prompt);
  auto moved_again_state = moved_again.save();
  if (!moved_again_result || !moved_again_state ||
      !std::equal(control_result->logits.begin(), control_result->logits.end(),
                  moved_again_result->logits.begin()) ||
      !equal_state(*control_state, *moved_again_state))
    return fail("prefill after session move with initialized workspace");
  std::cout << "authority primary-language decode, restore and replay passed\n";
  return 0;
}
