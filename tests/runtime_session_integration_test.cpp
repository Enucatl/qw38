#include "cuda/alloc.hpp"
#include "cuda/copy.hpp"
#include "format/format.hpp"
#include "runtime/runtime.hpp"
#include "runtime/sizes.hpp"
#include "runtime_support.hpp"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

using qw38::runtime::ErrorCode;
using qw38::runtime::kFixedPersistentBytes;
using qw38::runtime::kKvBytesPerToken;
using qw38::runtime::kMaxKvCapacity;
using qw38::runtime::Runtime;
using qw38::runtime::Session;
using qw38::runtime::test::write_language_fixture;

namespace {

int g_failures = 0;

void fail(std::string_view what) {
  std::cerr << "FAIL: " << what << '\n';
  ++g_failures;
}

void expect(bool cond, std::string_view what) {
  if (!cond) {
    fail(what);
  }
}

std::expected<Session, qw38::runtime::Error> create_session_that_outlives_runtime(
    std::filesystem::path const& model_path) {
  auto runtime = Runtime::create();
  if (!runtime) {
    return std::unexpected(runtime.error());
  }
  auto model = runtime->load(model_path);
  if (!model) {
    return std::unexpected(model.error());
  }
  return runtime->create_session(*model, 1);
}

}  // namespace

int main() {
  qw38::format::test::ScratchDir dir("qw38-runtime-int");
  auto fx = write_language_fixture(dir.file("model.qw38"));
  expect(!fx.path.empty(), "write language fixture");
  if (fx.path.empty()) {
    return 1;
  }

  auto rt = Runtime::create();
  expect(static_cast<bool>(rt), "Runtime::create");
  if (!rt) {
    return 1;
  }

  auto const mallocs_before_load = qw38::cuda::malloc_count();
  auto bad_path = dir.file("truncated.qw38");
  {
    std::ofstream out(bad_path, std::ios::binary);
    out.write("not-an-artifact", 15);
  }
  auto bad = rt->load(bad_path);
  expect(!bad && bad.error().code == ErrorCode::Format,
         "malformed artifact rejected");
  expect(qw38::cuda::malloc_count() == mallocs_before_load,
         "malformed artifact allocates no device memory");

  auto model = rt->load(fx.path);
  expect(static_cast<bool>(model), "upload fixture");
  if (!model) {
    std::cerr << qw38::runtime::error_message(model.error()) << '\n';
    return 1;
  }
  auto payload = model->payload("v");
  expect(static_cast<bool>(payload) && payload->pointer != nullptr &&
             !payload->writable && payload->rank == 1 && payload->extent[0] == 4,
         "immutable model tensor view");
  std::vector<std::byte> uploaded(8);
  expect(static_cast<bool>(
             qw38::cuda::copy_d2h(uploaded, payload->pointer, rt->stream())),
         "download uploaded tensor");
  expect(static_cast<bool>(rt->stream().sync()), "sync download");
  expect(uploaded == std::vector<std::byte>(fx.payload.begin(), fx.payload.end()),
         "uploaded bytes match fixture");

  constexpr std::uint64_t kCap = 8;
  auto const mallocs_before_sessions = qw38::cuda::malloc_count();
  auto zero_capacity = rt->create_session(*model, 0);
  expect(!zero_capacity &&
             zero_capacity.error().code == ErrorCode::InvalidCapacity,
         "zero capacity rejected before alloc");
  auto unsupported = rt->create_session(*model, kMaxKvCapacity + 1);
  expect(!unsupported &&
             unsupported.error().code == ErrorCode::InvalidCapacity,
         "one-past-maximum capacity rejected before alloc");
  expect(qw38::cuda::malloc_count() == mallocs_before_sessions,
         "invalid capacities do not allocate");

  auto s1 = rt->create_session(*model, kCap);
  auto s2 = rt->create_session(*model, kCap);
  expect(static_cast<bool>(s1) && static_cast<bool>(s2), "two sessions");
  if (!s1 || !s2) {
    if (!s1) {
      std::cerr << qw38::runtime::error_message(s1.error()) << '\n';
    }
    if (!s2) {
      std::cerr << qw38::runtime::error_message(s2.error()) << '\n';
    }
    return 1;
  }

  expect(s1->gdn_s().pointer != s2->gdn_s().pointer, "S buffers isolated");
  expect(s1->conv_history().pointer != s2->conv_history().pointer,
         "conv buffers isolated");
  expect(s1->kv().pointer != s2->kv().pointer, "KV buffers isolated");
  expect(s1->residual_h().pointer != s2->residual_h().pointer,
         "residual buffers isolated");
  expect(s1->persistent_bytes() == kFixedPersistentBytes + kKvBytesPerToken * kCap,
         "language-only persistent size");
  expect(s1->kv_capacity() == kCap && s1->kv_populated(0) == 0, "zero populated");
  expect(s1->gdn_s().extent[0] == 48 && s1->conv_history().extent[1] == 3 &&
             s1->kv().extent[3] == kCap,
         "state view extents");

  std::byte zero{};
  expect(static_cast<bool>(qw38::cuda::copy_d2h(
             &zero, s1->gdn_s().pointer, 1, rt->stream())),
         "read initial S");
  expect(static_cast<bool>(rt->stream().sync()), "sync zero check");
  expect(zero == std::byte{0}, "state is zero initially");

  expect(static_cast<bool>(
             qw38::cuda::fill_pattern(s1->gdn_s().pointer, 256, 0x11, rt->stream())),
         "fill session 1");
  expect(static_cast<bool>(
             qw38::cuda::fill_pattern(s2->gdn_s().pointer, 256, 0x22, rt->stream())),
         "fill session 2");
  expect(static_cast<bool>(rt->stream().sync()), "sync fills");
  std::byte b1{};
  std::byte b2{};
  expect(static_cast<bool>(qw38::cuda::copy_d2h(&b1, s1->gdn_s().pointer, 1,
                                                rt->stream())),
         "read s1");
  expect(static_cast<bool>(qw38::cuda::copy_d2h(&b2, s2->gdn_s().pointer, 1,
                                                rt->stream())),
         "read s2");
  expect(static_cast<bool>(rt->stream().sync()), "sync isolation read");
  expect(b1 == std::byte{0x11} && b2 == std::byte{0x22},
         "sessions do not share state");

  void const* s_addr = s1->gdn_s().pointer;
  void const* h_addr = s1->residual_h().pointer;
  void const* mid_addr = s1->residual_h_mid().pointer;
  auto scratch = s1->scratch(qw38::format::ScratchKind::NormalizedHidden);
  expect(static_cast<bool>(scratch), "normalized scratch");
  void const* scratch_addr = scratch ? scratch->pointer : nullptr;
  auto mixer = s1->scratch(qw38::format::ScratchKind::GdnWorkspace);
  auto attn = s1->scratch(qw38::format::ScratchKind::AttentionWorkspace);
  expect(mixer && attn && mixer->pointer == attn->pointer,
         "stable mixer reuse addresses");
  expect(scratch_addr != mixer->pointer, "normalized distinct from mixer");

  auto snap = s1->save();
  expect(static_cast<bool>(snap), "snapshot");
  if (snap) {
    expect(snap->gdn_s.size() == 150994944, "snapshot S bytes");
    expect(snap->conv_history.size() == 2949120, "snapshot conv bytes");
    expect(snap->kv.size() == kKvBytesPerToken * kCap, "snapshot KV bytes");
    expect(snap->gdn_s[0] == std::byte{0x11}, "snapshot captured pattern");
  }

  expect(!s1->set_populated_length(0, kCap + 1) &&
             s1->set_populated_length(0, kCap + 1).error().code ==
                 ErrorCode::InvalidPopulatedLength,
         "populated > capacity rejected");
  expect(static_cast<bool>(s1->set_populated_length(0, 3)), "populated within cap");
  expect(s1->kv_populated(0) == 3, "populated stored");

  auto cursor = s1->conv_cursor();
  cursor[1] = 2;
  expect(static_cast<bool>(s1->set_conv_cursor(cursor)), "set cursor");

  auto snap2 = s1->save();
  expect(static_cast<bool>(s1->reset()), "reset");
  expect(s1->kv_populated(0) == 0, "reset clears populated");
  expect(s1->conv_cursor()[1] == 0, "reset clears cursor");
  std::byte after_reset{};
  expect(static_cast<bool>(qw38::cuda::copy_d2h(&after_reset, s1->gdn_s().pointer,
                                                1, rt->stream())),
         "read after reset");
  expect(static_cast<bool>(rt->stream().sync()), "sync reset");
  expect(after_reset == std::byte{0}, "reset zeros S");
  expect(s1->gdn_s().pointer == s_addr && s1->residual_h().pointer == h_addr &&
             s1->residual_h_mid().pointer == mid_addr,
         "addresses stable across reset");
  auto scratch_after = s1->scratch(qw38::format::ScratchKind::NormalizedHidden);
  expect(scratch_after && scratch_after->pointer == scratch_addr,
         "scratch address stable across reset");

  if (snap2) {
    expect(static_cast<bool>(s1->restore(*snap2)), "restore");
    expect(s1->kv_populated(0) == 3, "restore populated");
    expect(s1->conv_cursor()[1] == 2, "restore cursor");
    std::byte restored{};
    expect(static_cast<bool>(qw38::cuda::copy_d2h(&restored, s1->gdn_s().pointer,
                                                  1, rt->stream())),
           "read restore");
    expect(static_cast<bool>(rt->stream().sync()), "sync restore");
    expect(restored == std::byte{0x11}, "restore exact S bytes");
    expect(s1->gdn_s().pointer == s_addr, "addresses stable across restore");
  }

  auto const mallocs_hot = qw38::cuda::malloc_count();
  expect(static_cast<bool>(s1->reset()), "hot reset");
  if (snap2) {
    expect(static_cast<bool>(s1->restore(*snap2)), "hot restore");
    auto snap3 = s1->save();
    expect(static_cast<bool>(snap3), "hot save");
  }
  expect(static_cast<bool>(s1->set_populated_length(0, 1)), "hot populated");
  (void)s1->gdn_s();
  (void)s1->residual_h();
  (void)s1->scratch(qw38::format::ScratchKind::MlpSwiglu);
  expect(static_cast<bool>(rt->stream().sync()), "hot sync");
  expect(qw38::cuda::malloc_count() == mallocs_hot,
         "no device allocations on session hot path");

  expect(s2->gdn_s().pointer != s1->gdn_s().pointer, "isolation after restore");
  std::byte still2{};
  expect(static_cast<bool>(qw38::cuda::copy_d2h(&still2, s2->gdn_s().pointer, 1,
                                                rt->stream())),
         "read s2 after s1 restore");
  expect(static_cast<bool>(rt->stream().sync()), "sync s2");
  expect(still2 == std::byte{0x22}, "session 2 unchanged");

  Runtime moved_runtime = std::move(*rt);
  expect(static_cast<bool>(moved_runtime.stream().sync()),
         "moved Runtime stream works");
  expect(static_cast<bool>(rt->stream().sync()),
         "moved-from Runtime stream remains valid");
  auto move_assignment_destination = Runtime::create();
  expect(static_cast<bool>(move_assignment_destination),
         "Runtime move-assignment destination");
  if (move_assignment_destination) {
    *move_assignment_destination = std::move(moved_runtime);
    expect(static_cast<bool>(move_assignment_destination->stream().sync()),
           "move-assigned Runtime stream works");
  }
  auto after_runtime_move = s1->save();
  expect(static_cast<bool>(after_runtime_move), "save after Runtime move");
  expect(static_cast<bool>(s1->reset()), "reset after Runtime move");
  if (after_runtime_move) {
    expect(static_cast<bool>(s1->restore(*after_runtime_move)),
           "restore after Runtime move");
  }

  auto surviving_session = create_session_that_outlives_runtime(fx.path);
  expect(static_cast<bool>(surviving_session), "create surviving session");
  if (surviving_session) {
    auto surviving_snapshot = surviving_session->save();
    expect(static_cast<bool>(surviving_snapshot),
           "save after Runtime destruction");
    expect(static_cast<bool>(surviving_session->reset()),
           "reset after Runtime destruction");
    if (surviving_snapshot) {
      expect(static_cast<bool>(surviving_session->restore(*surviving_snapshot)),
             "restore after Runtime destruction");
    }
    expect(surviving_session->gdn_s().pointer != nullptr &&
               surviving_session->conv_history().pointer != nullptr &&
               surviving_session->kv().pointer != nullptr &&
               surviving_session->residual_h().pointer != nullptr &&
               surviving_session->residual_h_mid().pointer != nullptr,
           "session views survive Runtime destruction");
    expect(static_cast<bool>(surviving_session->scratch(
               qw38::format::ScratchKind::NormalizedHidden)),
           "scratch survives Runtime destruction");
    expect(static_cast<bool>(surviving_session->set_populated_length(0, 1)),
           "population update survives Runtime destruction");
    auto populated = surviving_session->kv_populated_slot(0);
    expect(populated && **populated == 1,
           "population slot survives Runtime destruction");
    auto cursor = surviving_session->conv_cursor();
    cursor[0] = 2;
    expect(static_cast<bool>(surviving_session->set_conv_cursor(cursor)),
           "cursor update survives Runtime destruction");
    auto cursor_slot = surviving_session->conv_cursor_slot(0);
    expect(cursor_slot && **cursor_slot == 2,
           "cursor slot survives Runtime destruction");
  }

  if (g_failures != 0) {
    std::cerr << g_failures << " failures\n";
    return 1;
  }
  std::cout << "runtime_session_integration ok\n";
  return 0;
}
