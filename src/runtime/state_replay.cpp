#include "runtime/language_model.hpp"
#include "runtime/runtime.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <span>
#include <string>
#include <vector>

namespace {

struct Options {
  std::filesystem::path artifact, prompt, interleave, output;
};

bool options(int argc, char** argv, Options& out) {
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key == "--help") {
      std::cout << "usage: qw38-state-replay --artifact FILE --prompt U32LE "
                   "--interleave-prompt U32LE --output FILE\n";
      return false;
    }
    if (i + 1 >= argc) return false;
    std::filesystem::path value = argv[++i];
    if (key == "--artifact") out.artifact = value;
    else if (key == "--prompt") out.prompt = value;
    else if (key == "--interleave-prompt") out.interleave = value;
    else if (key == "--output") out.output = value;
    else return false;
  }
  return !out.artifact.empty() && !out.prompt.empty() &&
         !out.interleave.empty() && !out.output.empty();
}

bool read_ids(std::filesystem::path const& path, std::vector<std::uint32_t>& ids) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  std::vector<unsigned char> b((std::istreambuf_iterator<char>(in)), {});
  if (b.empty() || b.size() % 4) return false;
  ids.resize(b.size() / 4);
  for (std::size_t i = 0; i < ids.size(); ++i) {
    auto const p = 4 * i;
    ids[i] = std::uint32_t(b[p]) | (std::uint32_t(b[p + 1]) << 8u) |
             (std::uint32_t(b[p + 2]) << 16u) |
             (std::uint32_t(b[p + 3]) << 24u);
  }
  return true;
}

bool same(qw38::runtime::SessionSnapshot const& a,
          qw38::runtime::SessionSnapshot const& b) {
  return a.gdn_s == b.gdn_s && a.conv_history == b.conv_history && a.kv == b.kv &&
         a.conv_cursor == b.conv_cursor && a.gdn_position == b.gdn_position &&
         a.kv_populated == b.kv_populated && a.token_position == b.token_position;
}

std::vector<std::uint32_t> cycle(std::vector<std::uint32_t> const& source,
                                 std::size_t count) {
  std::vector<std::uint32_t> result;
  result.reserve(count);
  for (std::size_t i = 0; i < count; ++i) result.push_back(source[i % source.size()]);
  return result;
}

bool logits_equal(std::span<float const> a, std::span<float const> b) {
  return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

struct Continuation {
  std::vector<std::vector<float>> logits;
  std::vector<std::uint32_t> argmax;
  std::vector<qw38::runtime::SessionSnapshot> states;
};

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  if (!options(argc, argv, opt)) return 2;
  using namespace qw38::runtime;
  auto fail = [](Error const& error) {
    std::cerr << error_message(error) << '\n';
    return 1;
  };
  std::vector<std::uint32_t> source, other;
  if (!read_ids(opt.prompt, source) || !read_ids(opt.interleave, other)) {
    std::cerr << "invalid token input\n";
    return 1;
  }
  constexpr std::array<std::uint64_t, 9> checkpoints{1, 3, 4, 63, 64, 65, 255, 256, 257};
  auto const stream = cycle(source, 265);
  auto runtime = Runtime::create();
  if (!runtime) return fail(runtime.error());
  auto model = runtime->load(opt.artifact);
  if (!model) return fail(model.error());
  auto const& integrity = model->schema().integrity;
  if (integrity.empty()) return 1;
  std::ostringstream digest;
  for (auto byte : integrity.back().digest.bytes) {
    digest << std::hex << std::setw(2) << std::setfill('0')
           << static_cast<unsigned int>(byte);
  }
  std::vector<std::string> checkpoint_rows;
  auto write_status = [&](std::string const& status, std::string const& interleave) {
    auto temp = opt.output;
    temp += ".tmp";
    std::ofstream report(temp, std::ios::trunc);
    if (!report) return false;
    report << "{\"status\":\"" << status << "\",\"artifact_manifest_digest\":\""
           << digest.str() << "\",\"checkpoints\":[";
    for (std::size_t i = 0; i < checkpoint_rows.size(); ++i) {
      if (i) report << ',';
      report << checkpoint_rows[i];
    }
    report << "],\"interleave\":" << interleave << "}\n";
    report.close();
    if (!report) return false;
    std::error_code ec;
    std::filesystem::rename(temp, opt.output, ec);
    return !ec;
  };
  if (!write_status("INCOMPLETE", "null")) return 1;
  for (std::size_t ci = 0; ci < checkpoints.size(); ++ci) {
    auto const length = checkpoints[ci];
    auto const prefix = std::span<std::uint32_t const>(stream.data(), length);
    auto const tail = std::span<std::uint32_t const>(stream.data() + length, 8);
    auto baseline = runtime->create_session(*model, 265);
    if (!baseline) return fail(baseline.error());
    auto baseline_plan = LanguageModelPlan::bind(*model, *baseline, runtime->stream());
    if (!baseline_plan) return fail(baseline_plan.error());
    auto boundary = baseline_plan->setup_prompt_slow(prefix);
    if (!boundary) return fail(boundary.error());
    std::vector<float> boundary_logits(boundary->logits.begin(), boundary->logits.end());
    auto baseline_state = baseline->save();
    if (!baseline_state) return fail(baseline_state.error());
    Continuation expected;
    for (std::size_t i = 0; i < tail.size(); ++i) {
      auto result = baseline_plan->decode_token(tail[i], length + i);
      if (!result) return fail(result.error());
      expected.argmax.push_back(result->argmax);
      expected.logits.emplace_back(result->logits.begin(), result->logits.end());
      auto state = baseline->save();
      if (!state) return fail(state.error());
      expected.states.push_back(std::move(*state));
    }

    auto replay = runtime->create_session(*model, 265);
    if (!replay) return fail(replay.error());
    auto replay_plan = LanguageModelPlan::bind(*model, *replay, runtime->stream());
    if (!replay_plan) return fail(replay_plan.error());
    auto bprefix = std::span<std::uint32_t const>(other.data(), std::min<std::size_t>(other.size(), 16));
    auto b = replay_plan->setup_prompt_slow(bprefix);
    if (!b) return fail(b.error());
    if (auto reset = replay->reset(); !reset) return fail(reset.error());
    auto reset_boundary = replay_plan->setup_prompt_slow(prefix);
    if (!reset_boundary) return fail(reset_boundary.error());
    auto reset_state = replay->save();
    if (!reset_state) return fail(reset_state.error());
    bool reset_ok = logits_equal(boundary_logits, reset_boundary->logits) && same(*baseline_state, *reset_state);
    for (std::size_t i = 0; i < tail.size(); ++i) {
      auto result = replay_plan->decode_token(tail[i], length + i);
      auto state = replay->save();
      if (!result || !state) return result ? fail(state.error()) : fail(result.error());
      reset_ok = reset_ok && result->argmax == expected.argmax[i] && logits_equal(result->logits, expected.logits[i]) && same(*state, expected.states[i]);
    }

    auto restored = runtime->create_session(*model, 265);
    if (!restored) return fail(restored.error());
    if (auto result = restored->restore(*baseline_state); !result) return fail(result.error());
    auto restored_plan = LanguageModelPlan::bind(*model, *restored, runtime->stream());
    if (!restored_plan) return fail(restored_plan.error());
    bool snapshot_ok = true;
    for (std::size_t i = 0; i < tail.size(); ++i) {
      auto result = restored_plan->decode_token(tail[i], length + i);
      auto state = restored->save();
      if (!result || !state) return result ? fail(state.error()) : fail(result.error());
      snapshot_ok = snapshot_ok && result->argmax == expected.argmax[i] && logits_equal(result->logits, expected.logits[i]) && same(*state, expected.states[i]);
    }
    std::ostringstream row;
    row << "{\"length\":" << length << ",\"suffix_tokens\":8,\"reset_replay\":" << (reset_ok ? "true" : "false")
        << ",\"snapshot_restore\":" << (snapshot_ok ? "true" : "false") << '}';
    checkpoint_rows.push_back(row.str());
    bool const ok = reset_ok && snapshot_ok;
    if (!write_status(ok ? "INCOMPLETE" : "FAIL", "null")) return 1;
    std::cout << "checkpoint=" << length << " replay=" << (ok ? "PASS" : "FAIL") << '\n' << std::flush;
    if (!ok) return 1;
  }
  auto interleave = runtime->create_session(*model, std::max(source.size(), other.size()));
  if (!interleave) return fail(interleave.error());
  auto interleave_plan = LanguageModelPlan::bind(*model, *interleave, runtime->stream());
  if (!interleave_plan) return fail(interleave_plan.error());
  auto first_a = interleave_plan->setup_prompt_slow(source);
  if (!first_a) return fail(first_a.error());
  std::vector<float> first_a_logits(first_a->logits.begin(), first_a->logits.end());
  auto first_a_state = interleave->save();
  if (!first_a_state) return fail(first_a_state.error());
  if (auto result = interleave->reset(); !result) return fail(result.error());
  auto bresult = interleave_plan->setup_prompt_slow(other);
  if (!bresult) return fail(bresult.error());
  if (auto result = interleave->reset(); !result) return fail(result.error());
  auto second_a = interleave_plan->setup_prompt_slow(source);
  if (!second_a) return fail(second_a.error());
  auto second_a_state = interleave->save();
  if (!second_a_state) return fail(second_a_state.error());
  bool const interleave_ok = logits_equal(first_a_logits, second_a->logits) && same(*first_a_state, *second_a_state);
  std::ostringstream interleave_json;
  interleave_json << "{\"schedule\":\"A,reset,B,reset,A\",\"identical\":"
                  << (interleave_ok ? "true" : "false") << '}';
  if (!write_status(interleave_ok ? "PASS" : "FAIL", interleave_json.str())) return 1;
  if (!interleave_ok) return 1;
  return 0;
}
