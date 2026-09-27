#include "runtime/language_model.hpp"
#include "runtime/runtime.hpp"
#include "runtime/profiling.hpp"
#include "cuda/alloc.hpp"
#include "cuda/copy.hpp"

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
struct LanguageLayerPlanTestAccess {
  static void const* gamma(LanguageLayerPlan& layer, void const* pointer) {
    return std::exchange(layer.mlp_.gamma.pointer, pointer);
  }
  static void* set_first_output(LanguageLayerPlan& layer, void* pointer) {
    return std::exchange(layer.mlp_.next_h.pointer, pointer);
  }
};
struct LanguageModelPlanTestAccess {
  static auto capture(LanguageModelPlan& plan, unsigned token, std::uint64_t position) {
    auto pending = plan.pending(token, position, 1);
    if (!pending) return std::expected<void, Error>(std::unexpected(pending.error()));
    return plan.prepare_graph(*pending);
  }
  static auto builds(LanguageModelPlan const& p) { return p.graph_builds_; }
  static auto replays(LanguageModelPlan const& p) { return p.graph_replays_; }
  static bool uncommitted(LanguageModelPlan const& p, std::uint64_t position) {
    return p.state_->token_position() == position && p.state_->layers_at(position);
  }
  static void* first_output(LanguageModelPlan& p, void* pointer) {
    return LanguageLayerPlanTestAccess::set_first_output(p.layers_.front(), pointer);
  }

  static void const* first_gamma(LanguageModelPlan& p, void const* pointer) {
    p.graph_ = {}; // Changed immutable binding invalidates its captured addresses.
    return LanguageLayerPlanTestAccess::gamma(p.layers_.front(), pointer);
  }
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
  qw38::runtime::profiling::enabled = std::getenv("QW38_PROFILE") != nullptr;
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
  plan->set_decode_submission(DecodeSubmission::Eager);
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
  void const* requested_head_codes = nullptr;
  auto prefilled = plan->prefill_tokens(prompt, rows,
      [&](std::uint64_t row, std::span<float const> logits)
          -> std::expected<void, Error> {
        if (row != delivered || logits.size() != kVocab ||
            !std::ranges::all_of(logits, [](float x) { return std::isfinite(x); }))
          return std::unexpected(make_error(ErrorCode::Internal, "prefill.test",
                                            "requested row mismatch"));
        ++delivered;
        if (row == 1) {
          // A second final readout would now fail. Restore after the call.
          requested_head_codes = LanguageModelPlanTestAccess::prefill_head_codes(*plan);
          LanguageModelPlanTestAccess::set_prefill_head_codes(*plan, nullptr);
        }
        return {};
      });
  if (requested_head_codes)
    LanguageModelPlanTestAccess::set_prefill_head_codes(*plan, requested_head_codes);
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
      [&](std::uint64_t, std::span<float const>) -> std::expected<void, Error> {
        if (boundary_session->save() || boundary_session->reset())
          return std::unexpected(make_error(ErrorCode::Internal, "prefill.busy",
                                            "pending unit was externally observable"));
        return std::unexpected(make_error(ErrorCode::Internal, "prefill.test",
                                          "injected sink failure"));
      });
  if (!LanguageModelPlanTestAccess::uncommitted(*boundary_plan, 0) ||
      returned_failure || returned_failure.error().field != "prefill.test" ||
      boundary_session->save() ||
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
  if (!boundary_session->reset()) return fail("second chunk failure reset");
  std::array<std::uint64_t,1> second_chunk_row{256};
  auto second_chunk_failure = boundary_plan->prefill_tokens(sequence, second_chunk_row,
      [](std::uint64_t, std::span<float const>) -> std::expected<void, Error> {
        return std::unexpected(make_error(ErrorCode::Internal, "sink.chunk", "injected second chunk failure"));
      });
  if (second_chunk_failure || second_chunk_failure.error().field != "sink.chunk" ||
      !LanguageModelPlanTestAccess::uncommitted(*boundary_plan, 256) || boundary_session->save())
    return fail("failed second chunk preserves only the completed first chunk");
  if (!boundary_session->reset()) return fail("recover second chunk failure");
  auto graph_session = runtime->create_session(*model, 514);
  if (!graph_session) return fail("create graph session");
  auto graph_plan = LanguageModelPlan::bind(*model, *graph_session, runtime->stream());
  if (!graph_plan) return fail("bind graph session");
  using Access = LanguageModelPlanTestAccess;
  std::vector<std::uint32_t> graph_prefix(254);
  for (unsigned i = 0; i < graph_prefix.size(); ++i) graph_prefix[i] = 1 + i % 19;
  if (!graph_plan->prefill_tokens(graph_prefix)) return fail("graph prefill prefix");
  for (unsigned position : {254u,255u,256u,510u,511u,512u}) {
    if (position == 510) {
      std::vector<std::uint32_t> middle(253, 7);
      if (!graph_plan->prefill_tokens(middle)) return fail("graph second prefix");
    }
    auto before = graph_session->save();
    if (!before) return fail("graph save before");
    auto invalid = graph_plan->decode_token(kVocab, position);
    auto after_invalid = graph_session->save();
    if (invalid || !after_invalid || !equal_state(*before, *after_invalid))
      return fail("pre-enqueue rejection leaves bytes and counters unchanged");
    graph_plan->set_decode_submission(DecodeSubmission::Eager);
    auto syncs = qw38::cuda::testing::stream_sync_count();
    auto allocations = qw38::cuda::malloc_count();
    auto eager = graph_plan->decode_token(1 + position % 23, position);
    if (!eager || qw38::cuda::testing::stream_sync_count() != syncs + 1 ||
        qw38::cuda::malloc_count() != allocations) {
      if (!eager) std::cerr << error_message(eager.error()) << '\n';
      std::cerr << "sync delta=" << qw38::cuda::testing::stream_sync_count() - syncs
                << " allocation delta=" << qw38::cuda::malloc_count() - allocations << '\n';
      return fail("one eager completion and no hot allocation");
    }
    std::vector<float> expected_logits(eager->logits.begin(), eager->logits.end());
    auto expected_state = graph_session->save();
    if (!expected_state || !graph_session->restore(*before)) return fail("restore graph baseline");
    auto builds = Access::builds(*graph_plan);
    std::size_t free_before = 0, free_after = 0, total = 0;
    if (cudaMemGetInfo(&free_before, &total) != cudaSuccess) return fail("graph memory before");
    auto setup_allocations = qw38::cuda::malloc_count();
    if (auto st = Access::capture(*graph_plan, 1 + position % 23, position); !st) {
      std::cerr << error_message(st.error()) << '\n';
      return fail("capture graph without inference");
    }
    if (cudaMemGetInfo(&free_after, &total) != cudaSuccess) return fail("graph memory after");
    std::cout << "graph setup populated=" << position+1
              << " device_allocation_delta=" << qw38::cuda::malloc_count()-setup_allocations
              << " free_memory_delta=" << static_cast<std::int64_t>(free_after)-static_cast<std::int64_t>(free_before)
              << '\n';
    auto after_capture = graph_session->save();
    if (!after_capture || !equal_state(*before, *after_capture))
      return fail("capture alone changes neither counters nor persistent bytes");
    bool const new_bucket = position == 254 || position == 256 || position == 512;
    if (Access::builds(*graph_plan) != builds + (new_bucket ? 1 : 0))
      return fail("capture only at bucket transitions");
    graph_plan->set_decode_submission(DecodeSubmission::Graph);
    auto replays = Access::replays(*graph_plan);
    syncs = qw38::cuda::testing::stream_sync_count();
    allocations = qw38::cuda::malloc_count();
    auto replay = graph_plan->decode_token(1 + position % 23, position);
    if (!replay || graph_plan->graph_fallback() ||
        Access::replays(*graph_plan) != replays + 1 ||
        qw38::cuda::testing::stream_sync_count() != syncs + 1 ||
        qw38::cuda::malloc_count() != allocations ||
        !std::equal(replay->logits.begin(), replay->logits.end(), expected_logits.begin()))
      return fail("bitwise graph logits, one replay/completion, no hot allocation");
    auto actual_state = graph_session->save();
    if (!actual_state || !equal_state(*expected_state, *actual_state))
      return fail("bitwise graph state and all metadata at partition/bucket boundary");
    std::cout << "graph populated=" << position + 1 << " bitwise PASS\n";
  }
  // External capture must be rejected without poisoning or adding graph work.
  auto graph_before = graph_session->save();
  auto capture = qw38::cuda::Graph::begin(runtime->stream());
  if (!graph_before || !capture) return fail("external capture setup");
  auto captured_decode = graph_plan->decode_token(4, 513);
  std::array<std::uint32_t,1> last{4};
  auto captured_prefill = graph_plan->prefill_tokens(last);
  if (!capture->finish() || captured_decode || captured_prefill)
    return fail("external capture rejection");
  auto graph_after = graph_session->save();
  if (!graph_after || !equal_state(*graph_before, *graph_after))
    return fail("external capture leaves state usable");

  // Plan and session movement preserve the captured device addresses.
  LanguageModelPlan moved_graph_plan = std::move(*graph_plan);
  Session moved_graph_session = std::move(*graph_session);
  if (graph_plan->decode_token(4, 513)) return fail("moved-from plan rejection");
  // Deferred replay failure must not commit any counters or retry eagerly.
  auto replays = Access::replays(moved_graph_plan);
  qw38::cuda::testing::fail_next_stream_sync();
  auto late = moved_graph_plan.decode_token(4, 513);
  if (late || Access::replays(moved_graph_plan) != replays + 1 ||
      !Access::uncommitted(moved_graph_plan, 513) ||
      moved_graph_session.save() || moved_graph_plan.decode_token(4, 513))
    return fail("late graph failure poisons without commit or retry");
  if (!moved_graph_session.restore(*graph_before) ||
      !moved_graph_plan.decode_token(4, 513)) return fail("graph restore recovery");
  auto recovery = moved_graph_session.save();
  if (!recovery || !moved_graph_session.restore(*graph_before)) return fail("graph recovery baseline");
  moved_graph_plan.set_decode_submission(DecodeSubmission::Eager);
  if (!moved_graph_plan.decode_token(4, 513)) return fail("eager recovery reference");
  auto recovered_eager = moved_graph_session.save();
  if (!recovered_eager || !equal_state(*recovery, *recovered_eager))
    return fail("graph recovery exact state");
  if (!moved_graph_session.reset()) return fail("graph reset");
  auto clean_graph = moved_graph_session.save();
  auto first_output = Access::first_output(moved_graph_plan, nullptr);
  auto early = moved_graph_plan.decode_token(8, 0);
  Access::first_output(moved_graph_plan, first_output);
  if (early || !Access::uncommitted(moved_graph_plan, 0) ||
      moved_graph_session.save()) return fail("early mutation failure leaves counters uncommitted");
  auto mutated_history = clean_graph->conv_history;
  if (!qw38::cuda::copy_d2h(mutated_history.data(), moved_graph_session.conv_history().pointer,
                            mutated_history.size(), runtime->stream()) ||
      !runtime->stream().sync() || mutated_history == clean_graph->conv_history)
    return fail("early failure actually mutated device history");
  if (!moved_graph_session.restore(*clean_graph)) return fail("early restore");
  moved_graph_plan.set_decode_submission(DecodeSubmission::Graph);
  if (!moved_graph_plan.decode_token(8, 0)) return fail("reset refreshes controls and bucket");
  auto reset_reference = moved_graph_session.save();
  if (!moved_graph_session.reset() || !moved_graph_plan.decode_token(8, 0))
    return fail("reset graph replay");
  auto reset_replay = moved_graph_session.save();
  if (!reset_reference || !reset_replay || !equal_state(*reset_reference, *reset_replay))
    return fail("exact reset recovery");
  // A sticky producer error from layer zero must survive all later MLPs.
  auto bad_gamma = qw38::cuda::DeviceBuffer::allocate(kHidden * 2u);
  std::vector<std::uint16_t> infinity_gamma(kHidden, 0x7f80);
  if (!bad_gamma || !qw38::cuda::copy_h2d(bad_gamma->data(), infinity_gamma.data(),
        kHidden * 2u, runtime->stream()) || !runtime->stream().sync())
    return fail("producer failure fixture");
  for (auto mode : {DecodeSubmission::Eager, DecodeSubmission::Graph}) {
    if (!moved_graph_session.reset()) return fail("producer reset");
    auto original_gamma = Access::first_gamma(moved_graph_plan, bad_gamma->data());
    moved_graph_plan.set_decode_submission(mode);
    auto producer = moved_graph_plan.decode_token(8, 0);
    Access::first_gamma(moved_graph_plan, original_gamma);
    if (producer || producer.error().field != "q8.producer" ||
        producer.error().code != ErrorCode::InvalidArgument ||
        !Access::uncommitted(moved_graph_plan, 0) || moved_graph_session.save())
      return fail("sticky deferred producer error, typed original error and no commit");
  }
  if (!moved_graph_session.reset()) return fail("fallback reset");
  auto fallback_plan = LanguageModelPlan::bind(*model, moved_graph_session, runtime->stream());
  if (!fallback_plan) return fail("fallback rebind");
  fallback_plan->set_decode_submission(DecodeSubmission::Graph);
  qw38::cuda::testing::fail_next_graph_instantiation();
  auto fallback = fallback_plan->decode_token(8, 0);
  auto fallback_state = moved_graph_session.save();
  if (!fallback || !fallback_plan->graph_fallback() ||
      fallback_plan->graph_fallback()->field != "cudaGraphInstantiate" ||
      Access::replays(*fallback_plan) != 0 || !fallback_state ||
      !equal_state(*fallback_state, *reset_reference))
    return fail("pre-mutation construction fallback executes exactly once");
  std::cout << "construction fallback reason=" << error_message(*fallback_plan->graph_fallback()) << '\n';
  // Rebinding replaces the borrowed plan and invalidates all old graph geometry.
  auto rebound = LanguageModelPlan::bind(*model, moved_graph_session, runtime->stream());
  if (!rebound) return fail("fresh plan after fallback diagnostic");
  moved_graph_plan = std::move(*rebound);
  moved_graph_plan.set_decode_submission(DecodeSubmission::Graph);
  // Independent controls and scratch while interleaving plans on their session stream.
  if (!control_session->reset()) return fail("interleaved reset");
  auto independent = control_plan->decode_token(12, 0);
  std::vector<float> independent_logits;
  if (independent) independent_logits.assign(independent->logits.begin(), independent->logits.end());
  if (!independent || Access::replays(*control_plan) != 1 ||
      !moved_graph_plan.decode_token(9, 1) || !control_session->reset())
    return fail("interleaved graph tokens");
  auto independent_replay = control_plan->decode_token(12, 0);
  if (!independent_replay || control_plan->graph_fallback() ||
      !std::equal(independent_replay->logits.begin(), independent_replay->logits.end(),
                  independent_logits.begin()))
    return fail("session graphs do not share controls");

  std::cout << "TASK-039 graph/eager boundaries, capture, failures, movement and interleave passed\n";
  std::cout << "authority primary-language decode, restore and replay passed\n";
  return 0;
}
