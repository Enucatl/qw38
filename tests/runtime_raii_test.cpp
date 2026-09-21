#include "cuda/alloc.hpp"
#include "cuda/buffer.hpp"
#include "cuda/copy.hpp"
#include "cuda/error.hpp"
#include "cuda/event.hpp"
#include "cuda/stream.hpp"
#include "runtime/error.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using qw38::cuda::DeviceBuffer;
using qw38::cuda::ErrorCode;
using qw38::cuda::Event;
using qw38::cuda::from_status;
using qw38::cuda::Stream;

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

}  // namespace

int main() {
  auto translated = from_status(cudaErrorInvalidValue, "unit_translate");
  expect(translated.code == ErrorCode::Status, "status code");
  expect(translated.cuda_status == static_cast<int>(cudaErrorInvalidValue),
         "raw CUDA status preserved");
  expect(translated.operation == "unit_translate", "operation recorded");
  auto msg = qw38::cuda::error_message(translated);
  expect(msg.find("cuda_status") != std::string::npos, "typed error message");
  auto mapped = qw38::runtime::from_cuda(translated);
  expect(mapped.code == qw38::runtime::ErrorCode::Cuda, "runtime CUDA mapping");
  expect(mapped.cuda_status == static_cast<int>(cudaErrorInvalidValue),
         "runtime preserves cuda_status");

  auto stream = Stream::create();
  expect(static_cast<bool>(stream), "create ordered stream");
  if (!stream) {
    std::cerr << g_failures << " failures\n";
    return 1;
  }

  auto const mallocs0 = qw38::cuda::malloc_count();
  auto buf = DeviceBuffer::allocate(256);
  expect(static_cast<bool>(buf), "allocate 256 bytes");
  expect(qw38::cuda::malloc_count() == mallocs0 + 1, "malloc instrumentation");
  expect(!buf->empty() && buf->bytes() == 256, "buffer size");
  void* p = buf->data();

  DeviceBuffer moved = std::move(*buf);
  expect(buf->empty() && buf->data() == nullptr, "moved-from buffer is empty");
  expect(moved.data() == p && moved.bytes() == 256, "move steals pointer");

  {
    DeviceBuffer inner = std::move(moved);
    expect(inner.data() == p, "nested move");
  }
  expect(qw38::cuda::malloc_count() == mallocs0 + 1, "destructor does not malloc");
  expect(qw38::cuda::free_count() >= 1, "destructor frees");

  auto zero = DeviceBuffer::allocate(0);
  expect(!zero && zero.error().code == ErrorCode::InvalidArgument,
         "zero-byte alloc rejected");

  auto huge = DeviceBuffer::allocate(std::numeric_limits<std::uint64_t>::max());
  expect(!huge, "oversize alloc fails");
  if (!huge) {
    auto rt = qw38::runtime::from_cuda(huge.error());
    expect(rt.code == qw38::runtime::ErrorCode::Cuda ||
               rt.code == qw38::runtime::ErrorCode::AllocationFailed,
           "oversize maps to typed CUDA/allocation error");
    expect(rt.cuda_status != 0 || huge.error().code == ErrorCode::InvalidArgument,
           "CUDA status or typed invalid captured");
  }

  auto event = Event::create();
  expect(static_cast<bool>(event), "create event");
  if (event) {
    expect(static_cast<bool>(event->record(*stream)), "record event");
    expect(static_cast<bool>(event->sync()), "sync event");
  }

  Stream taken = std::move(*stream);
  expect(stream->empty(), "moved-from stream is empty");
  auto sync_empty = stream->sync();
  expect(!sync_empty && sync_empty.error().code == ErrorCode::InvalidArgument,
         "empty stream sync is a typed error");
  expect(static_cast<bool>(taken.sync()), "moved stream still works");

  Event ev2 = std::move(*event);
  expect(event->empty(), "moved-from event is empty");

  auto pattern_buf = DeviceBuffer::allocate(64);
  expect(static_cast<bool>(pattern_buf), "pattern buffer");
  if (pattern_buf) {
    expect(static_cast<bool>(qw38::cuda::fill_pattern(pattern_buf->data(), 64, 0x10,
                                                      taken)),
           "pattern fill");
    expect(static_cast<bool>(taken.sync()), "sync fill");
    std::vector<std::byte> host(64);
    expect(static_cast<bool>(
               qw38::cuda::copy_d2h(host, pattern_buf->data(), taken)),
           "copy D2H");
    expect(static_cast<bool>(taken.sync()), "sync D2H");
    expect(static_cast<unsigned>(host[0]) == 0x10, "pattern byte 0");
    expect(static_cast<unsigned>(host[5]) == 0x15, "pattern byte 5");
  }

  if (g_failures != 0) {
    std::cerr << g_failures << " failures\n";
    return 1;
  }
  std::cout << "runtime_raii ok\n";
  return 0;
}
