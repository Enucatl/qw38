#include "runtime/language_model.hpp"
#include "runtime/runtime.hpp"
#include "format/floatcvt.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
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
  std::vector<std::uint32_t> partitions;
  bool prefill{};
};

bool options(int argc, char** argv, Options& out) {
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key == "--help") {
      std::cout << "usage: qw38-state-replay --artifact FILE --prompt U32LE "
                   "--interleave-prompt U32LE --output FILE "
                   "[--prefill --partition N[,N]]\n";
      return false;
    }
    if (key == "--prefill") { out.prefill = true; continue; }
    if (i + 1 >= argc) return false;
    std::filesystem::path value = argv[++i];
    if (key == "--artifact") out.artifact = value;
    else if (key == "--prompt") out.prompt = value;
    else if (key == "--interleave-prompt") out.interleave = value;
    else if (key == "--output") out.output = value;
    else if (key == "--partition") {
      auto text = value.string();
      while (!text.empty()) {
        auto comma = text.find(',');
        auto part = text.substr(0, comma);
        std::uint32_t count{};
        auto [end, ec] = std::from_chars(part.data(), part.data() + part.size(), count);
        if (ec != std::errc{} || end != part.data() + part.size() ||
            count == 0 || count > 256) return false;
        out.partitions.push_back(count);
        if (comma == std::string::npos) break;
        text.erase(0, comma + 1);
        if (text.empty()) return false;
      }
    }
    else return false;
  }
  if (out.prefill && out.partitions.empty()) out.partitions.push_back(256);
  if (!out.prefill && !out.partitions.empty()) return false;
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

struct Difference {
  float max_abs{};
  std::uint64_t outside{}, nonfinite{};
};

Difference compare_f32(std::vector<std::byte> const& got,
                       std::vector<std::byte> const& reference,
                       float absolute, float relative) {
  Difference result;
  if (got.size() != reference.size() || got.size() % sizeof(float)) {
    result.nonfinite = 1;
    return result;
  }
  for (std::size_t i = 0; i < got.size(); i += sizeof(float)) {
    float a, b;
    std::memcpy(&a, got.data() + i, sizeof(float));
    std::memcpy(&b, reference.data() + i, sizeof(float));
    if (!std::isfinite(a) || !std::isfinite(b)) { ++result.nonfinite; continue; }
    auto const diff = std::fabs(a - b);
    result.max_abs = std::max(result.max_abs, diff);
    if (diff > absolute + relative * std::fabs(b)) ++result.outside;
  }
  return result;
}

Difference compare_bf16(std::vector<std::byte> const& got,
                        std::vector<std::byte> const& reference,
                        float absolute, float relative) {
  Difference result;
  if (got.size() != reference.size() || got.size() % sizeof(std::uint16_t)) {
    result.nonfinite = 1;
    return result;
  }
  for (std::size_t i = 0; i < got.size(); i += sizeof(std::uint16_t)) {
    std::uint16_t abits, bbits;
    std::memcpy(&abits, got.data() + i, sizeof(abits));
    std::memcpy(&bbits, reference.data() + i, sizeof(bbits));
    auto const a = qw38::format::bf16_to_fp32(abits);
    auto const b = qw38::format::bf16_to_fp32(bbits);
    if (!std::isfinite(a) || !std::isfinite(b)) { ++result.nonfinite; continue; }
    auto const diff = std::fabs(a - b);
    result.max_abs = std::max(result.max_abs, diff);
    if (diff > absolute + relative * std::fabs(b)) ++result.outside;
  }
  return result;
}

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
  auto setup = [&](LanguageModelPlan& plan, std::span<std::uint32_t const> ids)
      -> std::expected<DecodeResult, Error> {
    if (!opt.prefill) return plan.setup_prompt_slow(ids);
    std::expected<DecodeResult, Error> result = std::unexpected(
        make_error(ErrorCode::InvalidArgument, "prompt", "empty prompt"));
    for (std::size_t offset = 0, chunk = 0; offset < ids.size(); ++chunk) {
      auto count = std::min<std::size_t>(ids.size() - offset,
                                         opt.partitions[chunk % opt.partitions.size()]);
      result = plan.prefill_tokens(ids.subspan(offset, count));
      if (!result) return result;
      offset += count;
    }
    return result;
  };
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
           << digest.str() << "\",\"mode\":\""
           << (opt.prefill ? "prefill" : "decode") << "\",\"partitions\":[";
    for (std::size_t i = 0; i < opt.partitions.size(); ++i) {
      if (i) report << ',';
      report << opt.partitions[i];
    }
    report << "],\"checkpoints\":[";
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
    auto boundary = setup(*baseline_plan, prefix);
    if (!boundary) return fail(boundary.error());
    std::vector<float> boundary_logits(boundary->logits.begin(), boundary->logits.end());
    auto baseline_state = baseline->save();
    if (!baseline_state) return fail(baseline_state.error());
    std::string cross_json = "null";
    bool cross_ok = true;
    if (opt.prefill) {
      auto decoded = runtime->create_session(*model, 265);
      if (!decoded) return fail(decoded.error());
      auto decoded_plan = LanguageModelPlan::bind(*model, *decoded, runtime->stream());
      if (!decoded_plan) return fail(decoded_plan.error());
      auto decoded_boundary = decoded_plan->setup_prompt_slow(prefix);
      if (!decoded_boundary) return fail(decoded_boundary.error());
      auto decoded_state = decoded->save();
      if (!decoded_state) return fail(decoded_state.error());
      bool const metadata = baseline_state->token_position == decoded_state->token_position &&
          baseline_state->gdn_position == decoded_state->gdn_position &&
          baseline_state->kv_populated == decoded_state->kv_populated &&
          baseline_state->conv_cursor == decoded_state->conv_cursor;
      // Component thresholds are diagnostics here: 64-layer rounding can
      // compound, while local layer comparisons gate those thresholds.
      auto const gdn = compare_f32(baseline_state->gdn_s, decoded_state->gdn_s,
                                   8e-4f, 2e-4f);
      auto const history = compare_bf16(baseline_state->conv_history,
                                         decoded_state->conv_history, 0.02f, 0.006f);
      auto const kv = compare_bf16(baseline_state->kv, decoded_state->kv,
                                    0.005f, 0.002f);
      float logit_max = 0.0f;
      for (std::size_t i = 0; i < boundary_logits.size(); ++i)
        logit_max = std::max(logit_max,
                             std::fabs(boundary_logits[i] - decoded_boundary->logits[i]));
      cross_ok = metadata && !gdn.nonfinite && !history.nonfinite &&
                 !kv.nonfinite && std::isfinite(logit_max);
      std::ostringstream cross;
      auto number = [&](char const* name, Difference const& diff) {
        cross << "\"" << name << "\":{\"max_abs\":" << diff.max_abs
              << ",\"outside_component_tolerance\":" << diff.outside
              << ",\"nonfinite\":" << diff.nonfinite << '}';
      };
      cross << "{\"reference\":\"repeated_decode\",\"metadata_equal\":"
            << (metadata ? "true" : "false") << ",\"argmax_equal\":"
            << (boundary->argmax == decoded_boundary->argmax ? "true" : "false")
            << ",\"logit_max_abs\":" << logit_max << ',';
      number("gdn_state", gdn); cross << ',';
      number("conv_history", history); cross << ',';
      number("attention_kv", kv);
      cross << '}';
      cross_json = cross.str();
    }
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
    auto b = setup(*replay_plan, bprefix);
    if (!b) return fail(b.error());
    if (auto reset = replay->reset(); !reset) return fail(reset.error());
    auto reset_boundary = setup(*replay_plan, prefix);
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
        << ",\"snapshot_restore\":" << (snapshot_ok ? "true" : "false")
        << ",\"cross_schedule\":" << cross_json << '}';
    checkpoint_rows.push_back(row.str());
    bool const ok = reset_ok && snapshot_ok && cross_ok;
    if (!write_status(ok ? "INCOMPLETE" : "FAIL", "null")) return 1;
    std::cout << "checkpoint=" << length << " replay=" << (ok ? "PASS" : "FAIL") << '\n' << std::flush;
    if (!ok) return 1;
  }
  auto interleave = runtime->create_session(*model, std::max(source.size(), other.size()));
  if (!interleave) return fail(interleave.error());
  auto interleave_plan = LanguageModelPlan::bind(*model, *interleave, runtime->stream());
  if (!interleave_plan) return fail(interleave_plan.error());
  auto first_a = setup(*interleave_plan, source);
  if (!first_a) return fail(first_a.error());
  std::vector<float> first_a_logits(first_a->logits.begin(), first_a->logits.end());
  auto first_a_state = interleave->save();
  if (!first_a_state) return fail(first_a_state.error());
  if (auto result = interleave->reset(); !result) return fail(result.error());
  auto bresult = setup(*interleave_plan, other);
  if (!bresult) return fail(bresult.error());
  if (auto result = interleave->reset(); !result) return fail(result.error());
  auto second_a = setup(*interleave_plan, source);
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
