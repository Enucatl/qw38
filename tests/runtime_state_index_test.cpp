#include "cuda/copy.hpp"
#include "runtime/runtime.hpp"
#include "runtime/sizes.hpp"
#include "runtime_support.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string_view>
#include <vector>

using qw38::runtime::gdn_s_byte_offset;
using qw38::runtime::conv_history_byte_offset;
using qw38::runtime::kv_byte_offset;
using qw38::runtime::kConvTaps;
using qw38::runtime::Runtime;
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

}  // namespace

int main() {
  qw38::format::test::ScratchDir dir("qw38-state-index");
  auto fx = write_language_fixture(dir.file("model.qw38"));
  expect(!fx.path.empty(), "write fixture");
  if (fx.path.empty()) {
    return 1;
  }

  auto rt = Runtime::create();
  expect(static_cast<bool>(rt), "runtime create");
  if (!rt) {
    return 1;
  }
  auto model = rt->load(fx.path);
  expect(static_cast<bool>(model), "upload fixture");
  if (!model) {
    std::cerr << qw38::runtime::error_message(model.error()) << '\n';
    return 1;
  }
  constexpr std::uint64_t kCap = 8;
  auto session = rt->create_session(*model, kCap);
  expect(static_cast<bool>(session), "session");
  if (!session) {
    std::cerr << qw38::runtime::error_message(session.error()) << '\n';
    return 1;
  }

  float s_value = 42.0f;
  auto s_off = gdn_s_byte_offset(5, 7, 9, 11);
  expect(static_cast<bool>(s_off), "S offset");
  if (s_off) {
    expect(static_cast<bool>(qw38::cuda::copy_h2d(
               session->gdn_s().pointer
                   ? static_cast<std::byte*>(session->gdn_s().pointer) + *s_off
                   : nullptr,
               &s_value, sizeof(s_value), rt->stream())),
           "write S element");
    expect(static_cast<bool>(rt->stream().sync()), "sync S write");
    float back = 0;
    expect(static_cast<bool>(qw38::cuda::copy_d2h(
               &back,
               static_cast<std::byte*>(session->gdn_s().pointer) + *s_off,
               sizeof(back), rt->stream())),
           "read S element");
    expect(static_cast<bool>(rt->stream().sync()), "sync S read");
    expect(back == 42.0f, "S [layer,head,value,key] indexing");
  }

  std::uint16_t conv_bits = 0xABCD;
  auto c_off = conv_history_byte_offset(2, 1, 100);
  expect(static_cast<bool>(c_off), "conv offset");
  if (c_off) {
    expect(static_cast<bool>(qw38::cuda::copy_h2d(
               static_cast<std::byte*>(session->conv_history().pointer) + *c_off,
               &conv_bits, sizeof(conv_bits), rt->stream())),
           "write conv element");
    std::uint16_t conv_back = 0;
    expect(static_cast<bool>(qw38::cuda::copy_d2h(
               &conv_back,
               static_cast<std::byte*>(session->conv_history().pointer) + *c_off,
               sizeof(conv_back), rt->stream())),
           "read conv element");
    expect(static_cast<bool>(rt->stream().sync()), "sync conv");
    expect(conv_back == 0xABCD, "conv [layer,tap,channel] indexing");
  }

  std::uint16_t kv_bits = 0x1234;
  auto k_off = kv_byte_offset(3, 1, 2, 4, 8, kCap);
  expect(static_cast<bool>(k_off), "KV offset");
  if (k_off) {
    expect(static_cast<bool>(qw38::cuda::copy_h2d(
               static_cast<std::byte*>(session->kv().pointer) + *k_off, &kv_bits,
               sizeof(kv_bits), rt->stream())),
           "write KV element");
    std::uint16_t kv_back = 0;
    expect(static_cast<bool>(qw38::cuda::copy_d2h(
               &kv_back, static_cast<std::byte*>(session->kv().pointer) + *k_off,
               sizeof(kv_back), rt->stream())),
           "read KV element");
    expect(static_cast<bool>(rt->stream().sync()), "sync KV");
    expect(kv_back == 0x1234, "KV [layer,component,head,t,dim] indexing");
  }

  auto scratch = session->scratch(qw38::format::ScratchKind::GdnWorkspace);
  expect(static_cast<bool>(scratch), "GDN scratch view");
  if (scratch) {
    expect(static_cast<bool>(qw38::cuda::fill_pattern(scratch->pointer, 32, 0x40,
                                                      rt->stream())),
           "fill GDN scratch prefix");
    auto attn = session->scratch(qw38::format::ScratchKind::AttentionWorkspace);
    expect(static_cast<bool>(attn), "attention scratch view");
    if (attn) {
      expect(attn->pointer == scratch->pointer,
             "attention aliases mixer arena offset");
      std::vector<std::byte> host(8);
      expect(static_cast<bool>(
                 qw38::cuda::copy_d2h(host, attn->pointer, rt->stream())),
             "read aliased mixer bytes");
      expect(static_cast<bool>(rt->stream().sync()), "sync mixer alias");
      expect(static_cast<unsigned>(host[0]) == 0x40, "shared mixer pattern");
    }
  }

  // Device-to-device copy of a 16-byte S prefix into residual, then back.
  std::vector<std::byte> host16(16, std::byte{0x7E});
  expect(static_cast<bool>(qw38::cuda::copy_h2d(session->gdn_s().pointer, host16,
                                                rt->stream())),
         "pattern to S prefix");
  expect(static_cast<bool>(qw38::cuda::copy_d2d(
             session->residual_h().pointer, session->gdn_s().pointer, 16,
             rt->stream())),
         "D2D S -> residual");
  std::vector<std::byte> back16(16);
  expect(static_cast<bool>(qw38::cuda::copy_d2h(back16,
                                                session->residual_h().pointer,
                                                rt->stream())),
         "D2H residual");
  expect(static_cast<bool>(rt->stream().sync()), "sync D2D");
  expect(back16 == host16, "D2D copy preserves bytes");

  expect(session->conv_cursor().size() == 48, "per-layer conv cursor");
  auto cursor = session->conv_cursor();
  cursor[4] = 2;
  expect(static_cast<bool>(session->set_conv_cursor(cursor)), "set cursor");
  expect(session->conv_cursor()[4] == 2, "cursor persisted in session");
  cursor[0] = kConvTaps;
  expect(!session->set_conv_cursor(cursor), "cursor >= 3 rejected");

  if (g_failures != 0) {
    std::cerr << g_failures << " failures\n";
    return 1;
  }
  std::cout << "runtime_state_index ok\n";
  return 0;
}
