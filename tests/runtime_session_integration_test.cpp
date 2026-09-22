#include <algorithm>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "cuda/alloc.hpp"
#include "cuda/copy.hpp"
#include "format/format.hpp"
#include "runtime/runtime.hpp"
#include "runtime/sizes.hpp"
#include "runtime_support.hpp"

using qw38::runtime::ErrorCode;
using qw38::runtime::kFixedPersistentBytes;
using qw38::runtime::kKvBytesPerToken;
using qw38::runtime::kMaxKvCapacity;
using qw38::runtime::Runtime;
using qw38::runtime::Session;
using qw38::runtime::test::write_language_fixture;

static_assert(std::is_nothrow_move_constructible_v<Runtime>);
static_assert(std::is_nothrow_move_assignable_v<Runtime>);
static_assert(std::is_nothrow_move_constructible_v<Session>);
static_assert(std::is_nothrow_move_assignable_v<Session>);
static_assert(!std::is_convertible_v<qw38::runtime::ConstTensorView,
                                     qw38::runtime::TensorView>);
static_assert(!std::is_invocable_v<void (*)(qw38::runtime::TensorView),
                                   qw38::runtime::ConstTensorView>);

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

bool aligned_256(void const *pointer) {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) %
                                       qw38::format::kSpanAlignment ==
                                   0;
}

bool is_zero(std::span<std::byte const> bytes) {
  return std::ranges::all_of(
      bytes, [](std::byte value) { return value == std::byte{}; });
}

bool matches_pattern(std::span<std::byte const> bytes, std::uint8_t seed) {
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    auto const expected =
        static_cast<std::byte>(seed + static_cast<std::uint8_t>(i));
    if (bytes[i] != expected) {
      return false;
    }
  }
  return true;
}

struct SessionAddresses {
  void const *gdn_s{};
  void const *conv_history{};
  void const *kv{};
  void const *residual_h{};
  void const *residual_h_mid{};
  void const *normalized{};
  void const *mixer{};
};

SessionAddresses addresses(Session const &session) {
  auto normalized =
      session.scratch(qw38::format::ScratchKind::NormalizedHidden);
  auto mixer = session.scratch(qw38::format::ScratchKind::GdnWorkspace);
  return {
      .gdn_s = session.gdn_s().pointer,
      .conv_history = session.conv_history().pointer,
      .kv = session.kv().pointer,
      .residual_h = session.residual_h().pointer,
      .residual_h_mid = session.residual_h_mid().pointer,
      .normalized = normalized ? normalized->pointer : nullptr,
      .mixer = mixer ? mixer->pointer : nullptr,
  };
}

void expect_addresses(Session const &session, SessionAddresses const &expected,
                      std::string_view what) {
  auto actual = addresses(session);
  expect(actual.gdn_s == expected.gdn_s &&
             actual.conv_history == expected.conv_history &&
             actual.kv == expected.kv &&
             actual.residual_h == expected.residual_h &&
             actual.residual_h_mid == expected.residual_h_mid &&
             actual.normalized == expected.normalized &&
             actual.mixer == expected.mixer,
         what);
}

std::expected<Session, qw38::runtime::Error>
create_session_that_outlives_runtime(std::filesystem::path const &model_path) {
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

  auto fixture_bytes = qw38::format::test::read_all(fx.path);
  auto truncated_path = dir.file("truncated-fixture.qw38");
  fixture_bytes.resize(fixture_bytes.size() - 1);
  qw38::format::test::write_all(truncated_path, fixture_bytes);
  auto truncated = rt->load(truncated_path);
  expect(!truncated && truncated.error().code == ErrorCode::Format,
         "truncated complete artifact rejected");
  expect(qw38::cuda::malloc_count() == mallocs_before_load,
         "truncated complete artifact allocates no device memory");

  fixture_bytes = qw38::format::test::read_all(fx.path);
  fixture_bytes[0] ^= std::byte{0xff};
  auto corrupt_header_path = dir.file("corrupt-header.qw38");
  qw38::format::test::write_all(corrupt_header_path, fixture_bytes);
  auto corrupt_header = rt->load(corrupt_header_path);
  expect(!corrupt_header && corrupt_header.error().code == ErrorCode::Format,
         "corrupt full artifact rejected");
  expect(qw38::cuda::malloc_count() == mallocs_before_load,
         "corrupt full artifact allocates no device memory");

  auto incompatible_path = dir.file("fixed-capacity-state.qw38");
  auto incompatible_schema = qw38::format::test::base_schema();
  incompatible_schema.tensors.push_back(
      qw38::format::test::unplaced_bf16_vector(1, "v", 4));
  auto incompatible_state = qw38::runtime::language_persistent_schema();
  incompatible_state[2].declared_capacity = 1;
  incompatible_state[2].bytes_per_layer = 4096;
  incompatible_state[2].total_bytes =
      incompatible_state[2].bytes_per_token;
  incompatible_schema.state.assign(incompatible_state.begin(),
                                   incompatible_state.end());
  incompatible_schema.scratch = qw38::runtime::test::language_scratch();
  auto incompatible_writer =
      qw38::format::ArtifactWriter::create(incompatible_path,
                                           incompatible_schema);
  expect(static_cast<bool>(incompatible_writer),
         "create format-valid runtime-incompatible artifact");
  if (incompatible_writer) {
    expect(static_cast<bool>(incompatible_writer->write_span(
               "v", qw38::format::SpanKind::Payload, fx.payload)) &&
               static_cast<bool>(incompatible_writer->finalize()),
           "write format-valid runtime-incompatible artifact");
  }
  auto incompatible_artifact =
      qw38::format::Artifact::open(incompatible_path);
  expect(static_cast<bool>(incompatible_artifact),
         "runtime-incompatible artifact remains format-valid");
  if (incompatible_artifact) {
    auto incompatible = rt->upload(*incompatible_artifact);
    expect(!incompatible &&
               incompatible.error().code == ErrorCode::MalformedArtifact,
           "runtime-incompatible artifact rejected during upload preflight");
  }
  expect(qw38::cuda::malloc_count() == mallocs_before_load,
         "runtime-incompatible artifact allocates no device memory");

  auto model = rt->load(fx.path);
  expect(static_cast<bool>(model), "upload fixture");
  if (!model) {
    std::cerr << qw38::runtime::error_message(model.error()) << '\n';
    return 1;
  }
  auto payload = model->payload("v");
  expect(static_cast<bool>(payload) && payload->pointer != nullptr &&
             payload->rank == 1 && payload->extent[0] == 4,
         "immutable model tensor view");
  expect(payload && aligned_256(payload->pointer),
         "uploaded tensor is actually 256-byte aligned");
  std::vector<std::byte> uploaded(8);
  expect(static_cast<bool>(
             qw38::cuda::copy_d2h(uploaded, payload->pointer, rt->stream())),
         "download uploaded tensor");
  expect(static_cast<bool>(rt->stream().sync()), "sync download");
  expect(
      uploaded == std::vector<std::byte>(fx.payload.begin(), fx.payload.end()),
      "uploaded bytes match fixture");

  constexpr std::uint64_t kCap = 8;
  auto const mallocs_before_sessions = qw38::cuda::malloc_count();
  auto zero_capacity = rt->create_session(*model, 0);
  expect(!zero_capacity &&
             zero_capacity.error().code == ErrorCode::InvalidCapacity,
         "zero capacity rejected before alloc");
  auto unsupported = rt->create_session(*model, kMaxKvCapacity + 1);
  expect(!unsupported && unsupported.error().code == ErrorCode::InvalidCapacity,
         "one-past-maximum capacity rejected before alloc");
  expect(static_cast<bool>(qw38::runtime::validate_kv_capacity(1)),
         "minimum capacity boundary accepted");
  expect(static_cast<bool>(qw38::runtime::validate_kv_capacity(kMaxKvCapacity)),
         "maximum capacity boundary accepted without allocation");
  expect(qw38::cuda::malloc_count() == mallocs_before_sessions,
         "invalid capacities do not allocate");

  int device_count = 0;
  expect(cudaGetDeviceCount(&device_count) == cudaSuccess, "query device count");
  int const runtime_device = rt->device();
  int const caller_device =
      device_count > 1 ? (runtime_device + 1) % device_count : runtime_device;
  expect(cudaSetDevice(caller_device) == cudaSuccess,
         "select caller device before session creation");
  {
    auto minimum_capacity = rt->create_session(*model, 1);
    expect(static_cast<bool>(minimum_capacity),
           "minimum-capacity session can be allocated");
    if (minimum_capacity) {
      expect(minimum_capacity->kv_capacity() == 1 &&
                 minimum_capacity->kv().extent[3] == 1,
             "minimum-capacity session preserves boundary");
      expect(static_cast<bool>(minimum_capacity->reset()),
             "session operation uses runtime device");
    }
  }
  int current_device = -1;
  expect(cudaGetDevice(&current_device) == cudaSuccess &&
             current_device == caller_device,
         "runtime operations preserve caller device");
  expect(cudaSetDevice(runtime_device) == cudaSuccess,
         "restore runtime device");

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
  expect(
      s1->persistent_bytes() == kFixedPersistentBytes + kKvBytesPerToken * kCap,
      "language-only persistent size");
  expect(s1->kv_capacity() == kCap && s1->kv_populated(0) == 0,
         "zero populated");
  expect(s1->gdn_s().extent[0] == 48 && s1->conv_history().extent[1] == 3 &&
             s1->kv().extent[3] == kCap,
         "state view extents");
  auto stable_addresses = addresses(*s1);
  expect(aligned_256(stable_addresses.gdn_s) &&
             aligned_256(stable_addresses.conv_history) &&
             aligned_256(stable_addresses.kv) &&
             aligned_256(stable_addresses.residual_h) &&
             aligned_256(stable_addresses.residual_h_mid) &&
             aligned_256(stable_addresses.normalized) &&
             aligned_256(stable_addresses.mixer),
         "all session allocations and arena views are 256-byte aligned");

  std::byte zero{};
  expect(static_cast<bool>(
             qw38::cuda::copy_d2h(&zero, s1->gdn_s().pointer, 1, rt->stream())),
         "read initial S");
  expect(static_cast<bool>(rt->stream().sync()), "sync zero check");
  expect(zero == std::byte{0}, "state is zero initially");

  expect(
      static_cast<bool>(qw38::cuda::fill_pattern(
          s1->gdn_s().pointer, qw38::runtime::kGdnSBytes, 0x11, rt->stream())),
      "fill all session 1 S bytes");
  expect(static_cast<bool>(qw38::cuda::fill_pattern(
             s2->gdn_s().pointer, qw38::runtime::kGdnSBytes, 0x22,
             rt->stream())),
         "fill all session 2 S bytes");
  expect(static_cast<bool>(qw38::cuda::fill_pattern(
             s1->conv_history().pointer, qw38::runtime::kConvHistoryBytes, 0x33,
             rt->stream())),
         "fill all session 1 history bytes");
  expect(static_cast<bool>(qw38::cuda::fill_pattern(
             s1->kv().pointer, kKvBytesPerToken * kCap, 0x55, rt->stream())),
         "fill all session 1 KV bytes");
  expect(static_cast<bool>(rt->stream().sync()), "sync fills");
  std::byte b1{};
  std::byte b2{};
  expect(static_cast<bool>(
             qw38::cuda::copy_d2h(&b1, s1->gdn_s().pointer, 1, rt->stream())),
         "read s1");
  expect(static_cast<bool>(
             qw38::cuda::copy_d2h(&b2, s2->gdn_s().pointer, 1, rt->stream())),
         "read s2");
  expect(static_cast<bool>(rt->stream().sync()), "sync isolation read");
  expect(b1 == std::byte{0x11} && b2 == std::byte{0x22},
         "sessions do not share state");

  auto scratch = s1->scratch(qw38::format::ScratchKind::NormalizedHidden);
  expect(static_cast<bool>(scratch), "normalized scratch");
  void const *scratch_addr = scratch ? scratch->pointer : nullptr;
  auto mixer = s1->scratch(qw38::format::ScratchKind::GdnWorkspace);
  auto attn = s1->scratch(qw38::format::ScratchKind::AttentionWorkspace);
  expect(mixer && attn && mixer->pointer == attn->pointer,
         "stable mixer reuse addresses");
  expect(
      mixer && mixer->region_count == 11 &&
          mixer->region[0].offset == qw38::runtime::kGdnOffQkv &&
          mixer->region[0].bytes == qw38::runtime::kGdnBytesQkv &&
          mixer->region[0].stride_bytes ==
              qw38::runtime::kGdnWorkspaceBytesPerToken &&
          mixer->region[0].repetitions == qw38::runtime::kArenaTokenCapacity &&
          mixer->region[0].tensor.dtype ==
              qw38::format::ArithmeticDtype::Bf16 &&
          mixer->region[3].tensor.dtype == qw38::format::ArithmeticDtype::Fp32,
      "GDN workspace reports mixed typed regions");
  expect(
      attn && attn->region_count == 6 &&
          attn->region[5].offset == qw38::runtime::kAttnOffPartials &&
          attn->region[0].tensor.dtype == qw38::format::ArithmeticDtype::Bf16 &&
          attn->region[5].tensor.dtype == qw38::format::ArithmeticDtype::Fp32,
      "attention workspace reports mixed typed regions");
  expect(scratch_addr != mixer->pointer, "normalized distinct from mixer");

  expect(!s1->set_populated_length(0, kCap + 1) &&
             s1->set_populated_length(0, kCap + 1).error().code ==
                 ErrorCode::InvalidPopulatedLength,
         "populated > capacity rejected");
  expect(static_cast<bool>(s1->set_populated_length(0, 3)),
         "populated within cap");
  expect(s1->kv_populated(0) == 3, "populated stored");

  auto cursor = s1->conv_cursor();
  cursor[1] = 2;
  expect(static_cast<bool>(s1->set_conv_cursor(cursor)), "set cursor");

  auto snap2 = s1->save();
  expect(static_cast<bool>(snap2), "snapshot all persistent state");
  if (snap2) {
    expect(snap2->gdn_s ==
               qw38::format::test::pattern(qw38::runtime::kGdnSBytes, 0x11),
           "snapshot captures every S byte");
    expect(snap2->conv_history == qw38::format::test::pattern(
                                      qw38::runtime::kConvHistoryBytes, 0x33),
           "snapshot captures every history byte");
    expect(
        snap2->kv == qw38::format::test::pattern(kKvBytesPerToken * kCap, 0x55),
        "snapshot captures every KV byte");
    auto invalid_populated = *snap2;
    invalid_populated.kv_populated[0] = kCap + 1;
    auto bad_populated_restore = s1->restore(invalid_populated);
    expect(!bad_populated_restore && bad_populated_restore.error().code ==
                                         ErrorCode::InvalidPopulatedLength,
           "restore rejects populated length beyond capacity");
    auto invalid_cursor = *snap2;
    invalid_cursor.conv_cursor[0] = qw38::runtime::kConvTaps;
    auto bad_cursor_restore = s1->restore(invalid_cursor);
    expect(!bad_cursor_restore &&
               bad_cursor_restore.error().code == ErrorCode::InvalidArgument,
           "restore rejects invalid convolution cursor");
    expect(s1->kv_populated(0) == 3 && s1->conv_cursor()[1] == 2,
           "rejected restore leaves metadata unchanged");
  }
  qw38::cuda::testing::fail_next_stream_sync();
  auto failed_reset = s1->reset();
  expect(!failed_reset && failed_reset.error().code == ErrorCode::Cuda,
         "injected deferred reset failure is reported");
  expect(s1->kv_populated(0) == 3 && s1->conv_cursor()[1] == 2,
         "failed reset does not commit host metadata");
  expect(static_cast<bool>(s1->reset()), "reset");
  expect(s1->kv_populated(0) == 0, "reset clears populated");
  expect(s1->conv_cursor()[1] == 0, "reset clears cursor");
  expect_addresses(*s1, stable_addresses, "all addresses stable across reset");
  {
    auto reset_snapshot = s1->save();
    expect(static_cast<bool>(reset_snapshot), "snapshot reset state");
    if (reset_snapshot) {
      expect(is_zero(reset_snapshot->gdn_s), "reset zeros every S byte");
      expect(is_zero(reset_snapshot->conv_history),
             "reset zeros every history byte");
      expect(is_zero(reset_snapshot->kv), "reset zeros every KV byte");
    }
  }

  if (snap2) {
    qw38::cuda::testing::fail_next_stream_sync();
    auto failed_restore = s1->restore(*snap2);
    expect(!failed_restore && failed_restore.error().code == ErrorCode::Cuda,
           "injected deferred restore failure is reported");
    expect(s1->kv_populated(0) == 0 && s1->conv_cursor()[1] == 0,
           "failed restore does not commit host metadata");
    expect(static_cast<bool>(s1->restore(*snap2)), "restore");
    expect(s1->kv_populated(0) == 3, "restore populated");
    expect(s1->conv_cursor()[1] == 2, "restore cursor");
    auto restored = s1->save();
    expect(static_cast<bool>(restored), "snapshot restored state");
    if (restored) {
      expect(restored->gdn_s == snap2->gdn_s,
             "restore round-trips every S byte");
      expect(restored->conv_history == snap2->conv_history,
             "restore round-trips every history byte");
      expect(restored->kv == snap2->kv, "restore round-trips every KV byte");
    }
    expect_addresses(*s1, stable_addresses,
                     "all addresses stable across restore");
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
  {
    std::vector<std::byte> still2(qw38::runtime::kGdnSBytes);
    expect(static_cast<bool>(qw38::cuda::copy_d2h(
               still2, s2->gdn_s().pointer, rt->stream())),
           "read complete s2 S after s1 restore");
    expect(static_cast<bool>(rt->stream().sync()), "sync complete s2 S");
    expect(matches_pattern(still2, 0x22),
           "every session 2 S byte unchanged");
  }

  {
    auto const moved_addresses = addresses(*s2);
    Session moved_session = std::move(*s2);
    expect(s2->gdn_s().pointer == nullptr &&
               s2->conv_history().pointer == nullptr &&
               s2->kv().pointer == nullptr,
           "move-constructed Session relinquishes persistent buffers");
    expect_addresses(moved_session, moved_addresses,
                     "Session move construction preserves every address");
    auto moved_snapshot = moved_session.save();
    expect(static_cast<bool>(moved_snapshot),
           "move-constructed Session remains usable");

    auto assignment_destination = rt->create_session(*model, 1);
    expect(static_cast<bool>(assignment_destination),
           "Session move-assignment destination");
    if (assignment_destination) {
      auto const frees_before_assignment = qw38::cuda::free_count();
      *assignment_destination = std::move(moved_session);
      expect(moved_session.gdn_s().pointer == nullptr &&
                 moved_session.conv_history().pointer == nullptr &&
                 moved_session.kv().pointer == nullptr,
             "move-assigned Session relinquishes persistent buffers");
      expect_addresses(*assignment_destination, moved_addresses,
                       "Session move assignment preserves every address");
      expect(qw38::cuda::free_count() > frees_before_assignment,
             "Session move assignment destroys prior owned buffers");
      expect(static_cast<bool>(assignment_destination->reset()),
             "move-assigned Session remains usable");
    }
  }

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
    expect(surviving_session->kv_populated(0) == 1,
           "population metadata survives Runtime destruction");
    auto cursor = surviving_session->conv_cursor();
    cursor[0] = 2;
    expect(static_cast<bool>(surviving_session->set_conv_cursor(cursor)),
           "cursor update survives Runtime destruction");
    expect(surviving_session->conv_cursor()[0] == 2,
           "cursor metadata survives Runtime destruction");
  }

  {
    auto lifetime_runtime = Runtime::create();
    expect(static_cast<bool>(lifetime_runtime),
           "create Runtime for Session-first destruction");
    if (lifetime_runtime) {
      auto lifetime_model = lifetime_runtime->load(fx.path);
      expect(static_cast<bool>(lifetime_model),
             "load model for Session-first destruction");
      if (lifetime_model) {
        auto const bytes_before_session = qw38::cuda::live_bytes();
        {
          auto short_session =
              lifetime_runtime->create_session(*lifetime_model, 1);
          expect(static_cast<bool>(short_session),
                 "create short-lived Session");
          expect(qw38::cuda::live_bytes() > bytes_before_session,
                 "short-lived Session owns device allocations");
        }
        expect(qw38::cuda::live_bytes() == bytes_before_session,
               "Session destruction releases allocations before Runtime");
        expect(static_cast<bool>(lifetime_runtime->stream().sync()),
               "Runtime remains usable after Session destruction");
      }
    }
  }

  if (g_failures != 0) {
    std::cerr << g_failures << " failures\n";
    return 1;
  }
  std::cout << "runtime_session_integration ok\n";
  return 0;
}
