#include "format/reader.hpp"
#include "runtime/arena.hpp"
#include "runtime/runtime.hpp"
#include "runtime_support.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <new>
#include <string_view>

namespace {

std::atomic<bool> g_fail_next_allocation{false};

int g_failures = 0;

void fail(std::string_view what) {
  std::cerr << "FAIL: " << what << '\n';
  ++g_failures;
}

void expect(bool condition, std::string_view what) {
  if (!condition) {
    fail(what);
  }
}

void fail_next_host_allocation() noexcept {
  g_fail_next_allocation.store(true, std::memory_order_relaxed);
}

} // namespace

void *operator new(std::size_t bytes) {
  if (g_fail_next_allocation.exchange(false, std::memory_order_relaxed)) {
    throw std::bad_alloc();
  }
  if (void *p = std::malloc(bytes == 0 ? 1 : bytes)) {
    return p;
  }
  throw std::bad_alloc();
}

void operator delete(void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }

int main() {
  using qw38::runtime::ErrorCode;

  qw38::runtime::ScratchRequest request{
      .kind = qw38::format::ScratchKind::NormalizedHidden,
      .dtype = qw38::format::ArithmeticDtype::Bf16,
      .bytes = 256,
      .live = {{qw38::runtime::KernelStage::LayerNorm,
                qw38::runtime::KernelStage::LayerNorm}},
  };
  auto const requests = std::array{request};
  fail_next_host_allocation();
  auto arena = qw38::runtime::plan_scratch_arena(requests);
  expect(!arena && arena.error().code == ErrorCode::AllocationFailed,
         "arena allocation failure is typed");

  qw38::format::test::ScratchDir dir("qw38-runtime-allocation-failure");
  auto fixture =
      qw38::runtime::test::write_language_fixture(dir.file("model.qw38"));
  expect(!fixture.path.empty(), "write fixture");
  if (fixture.path.empty()) {
    return 1;
  }
  auto artifact = qw38::format::Artifact::open(fixture.path);
  expect(static_cast<bool>(artifact), "open fixture");
  auto runtime = qw38::runtime::Runtime::create();
  expect(static_cast<bool>(runtime), "create runtime");
  if (!artifact || !runtime) {
    return 1;
  }

  fail_next_host_allocation();
  auto failed_upload = runtime->upload(*artifact);
  expect(!failed_upload &&
             failed_upload.error().code == ErrorCode::AllocationFailed,
         "upload allocation failure is typed");

  auto model = runtime->upload(*artifact);
  expect(static_cast<bool>(model), "upload fixture");
  if (!model) {
    return 1;
  }

  fail_next_host_allocation();
  auto failed_session = runtime->create_session(*model, 1);
  expect(!failed_session &&
             failed_session.error().code == ErrorCode::AllocationFailed,
         "session creation allocation failure is typed");

  auto session = runtime->create_session(*model, 1);
  expect(static_cast<bool>(session), "create session");
  if (session) {
    fail_next_host_allocation();
    auto snapshot = session->save();
    expect(!snapshot && snapshot.error().code == ErrorCode::AllocationFailed,
           "snapshot allocation failure is typed");
  }

  if (g_failures != 0) {
    std::cerr << g_failures << " failures\n";
    return 1;
  }
  std::cout << "runtime_allocation_failure ok\n";
  return 0;
}
