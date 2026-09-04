#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "qw38/engine.h"
#include "response_store.h"
#include "responses_api.h"
#include "server_api.h"
#include "server_core.h"
#include "server_generation.h"
#include "server_json.h"

namespace {

constexpr const char* kBrand =
    "Quartz Watch 38 — Swiss-made inference for Qwen3.8-27B. It ticks fast.";
constexpr const char* kModelId = "qwen3.8-27b-q4_k_m";
constexpr std::size_t kMaximumHeaderBytes = 64 * 1024;
constexpr std::size_t kMaximumBodyBytes = 1024 * 1024;

volatile std::sig_atomic_t g_stop = 0;
volatile std::sig_atomic_t g_listener = -1;
std::atomic<std::uint64_t> g_response_counter{1};

struct Options final {
  std::string model;
  std::string host = "127.0.0.1";
  std::uint16_t port = 8080;
  std::string response_dir = "checkpoints/responses";
};

struct Request final {
  std::string method;
  std::string target;
  std::map<std::string, std::string> headers;
  std::string body;
};

struct Response final {
  int status = 200;
  std::string body;
  std::string extra_headers;
};

struct RouteContext final {
  const qw38::Engine* engine = nullptr;
  qw38::Session* session = nullptr;
  qw38::Token end_token = 0;
  qw38::server::SingleFlightGate* gate = nullptr;
  std::atomic<std::uint64_t>* cancelled_requests = nullptr;
  std::string model_id = kModelId;
  qw38::server::ResponseStore* response_store = nullptr;
};

class ConnectionTracker final {
 public:
  void start() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    ++active_;
  }

  void finish() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    --active_;
    changed_.notify_all();
  }

  void wait() noexcept {
    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait(lock, [this] { return active_ == 0; });
  }

 private:
  std::mutex mutex_;
  std::condition_variable changed_;
  std::size_t active_ = 0;
};

class ConnectionOwner final {
 public:
  ConnectionOwner(int socket, ConnectionTracker* tracker) noexcept
      : socket_(socket), tracker_(tracker) {}
  ~ConnectionOwner() {
    close(socket_);
    tracker_->finish();
  }

 private:
  int socket_ = -1;
  ConnectionTracker* tracker_ = nullptr;
};

class DisconnectMonitor final {
 public:
  explicit DisconnectMonitor(int socket) : socket_(socket) {
    worker_ = std::thread([this] { run(); });
  }

  ~DisconnectMonitor() { stop(); }
  DisconnectMonitor(const DisconnectMonitor&) = delete;
  DisconnectMonitor& operator=(const DisconnectMonitor&) = delete;

  std::atomic<bool>* cancelled() noexcept { return &cancelled_; }

  void stop() noexcept {
    stopped_.store(true, std::memory_order_relaxed);
    if (worker_.joinable()) worker_.join();
  }

 private:
  void run() noexcept {
    while (!stopped_.load(std::memory_order_relaxed)) {
      if (g_stop) {
        cancelled_.store(true, std::memory_order_relaxed);
        return;
      }
      pollfd descriptor{socket_, POLLIN, 0};
      const int result = poll(&descriptor, 1, 10);
      if (result < 0 && errno == EINTR) continue;
      if (result < 0 ||
          (descriptor.revents & (POLLHUP | POLLERR | POLLNVAL)) != 0) {
        cancelled_.store(true, std::memory_order_relaxed);
        return;
      }
      if (result > 0 && (descriptor.revents & POLLIN) != 0) {
        char byte = 0;
        const ssize_t count = recv(socket_, &byte, 1, MSG_PEEK | MSG_DONTWAIT);
        if (count == 0) {
          cancelled_.store(true, std::memory_order_relaxed);
          return;
        }
      }
    }
  }

  int socket_ = -1;
  std::atomic<bool> stopped_{false};
  std::atomic<bool> cancelled_{false};
  std::thread worker_;
};

void usage(std::ostream& output) {
  output << "usage: qw38-server MODEL [--host IPV4] [--port PORT] [--response-dir DIR]\n"
         << "  --host IPV4   listen address (default 127.0.0.1)\n"
         << "  --port PORT   TCP port; 0 asks the OS to choose (default 8080)\n"
         << "  --response-dir DIR  atomic continuation records (default checkpoints/responses)\n";
}

bool parse_port(const char* text, std::uint16_t* port) {
  if (text == nullptr || *text == '\0') return false;
  errno = 0;
  char* end = nullptr;
  const unsigned long value = std::strtoul(text, &end, 10);
  if (errno != 0 || *end != '\0' || value > 65535) return false;
  *port = static_cast<std::uint16_t>(value);
  return true;
}

bool parse_options(int argc, char** argv, Options* options) {
  if (argc < 2) return false;
  if (argc == 2 && std::string(argv[1]) == "--help") {
    usage(std::cout);
    std::exit(0);
  }
  options->model = argv[1];
  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help") {
      usage(std::cout);
      std::exit(0);
    }
    if (index + 1 >= argc) return false;
    const char* value = argv[++index];
    if (argument == "--host") options->host = value;
    else if (argument == "--response-dir") options->response_dir = value;
    else if (argument == "--port") {
      if (!parse_port(value, &options->port)) return false;
    } else {
      return false;
    }
  }
  return !options->host.empty() && !options->response_dir.empty();
}

void stop_server(int) {
  g_stop = 1;
  const int listener = g_listener;
  g_listener = -1;
  if (listener >= 0) close(listener);
}

const char* reason(int status) {
  if (status == 200) return "OK";
  if (status == 400) return "Bad Request";
  if (status == 404) return "Not Found";
  if (status == 405) return "Method Not Allowed";
  if (status == 411) return "Length Required";
  if (status == 413) return "Content Too Large";
  if (status == 499) return "Client Closed Request";
  return "Internal Server Error";
}

std::string error_body(const std::string& message, const std::string& code) {
  return std::string("{\"error\":{\"message\":") +
         qw38::server::quote_json(message) +
         ",\"type\":\"invalid_request_error\",\"code\":" +
         qw38::server::quote_json(code) + "}}";
}

bool send_all(int socket, const std::string& bytes) noexcept {
  std::size_t sent = 0;
  while (sent < bytes.size()) {
    const ssize_t count =
        send(socket, bytes.data() + sent, bytes.size() - sent, MSG_NOSIGNAL);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) return false;
    sent += static_cast<std::size_t>(count);
  }
  return true;
}

bool write_headers(int socket, int status, const std::string& content_type,
                   const std::string& extra_headers,
                   const std::string* body) noexcept {
  std::string header = "HTTP/1.1 " + std::to_string(status) + " " +
                       reason(status) + "\r\n";
  header += "Content-Type: " + content_type + "\r\n";
  if (body != nullptr) {
    header += "Content-Length: " + std::to_string(body->size()) + "\r\n";
  }
  header += "Connection: close\r\n";
  header += "Server: qw38\r\n";
  header += extra_headers;
  header += "\r\n";
  return send_all(socket, header);
}

void write_response(int socket, const Response& response) noexcept {
  if (write_headers(socket, response.status, "application/json",
                    response.extra_headers, &response.body)) {
    send_all(socket, response.body);
  }
}

std::string lower_ascii(std::string value) {
  for (char& byte : value) {
    if (byte >= 'A' && byte <= 'Z') byte = static_cast<char>(byte - 'A' + 'a');
  }
  return value;
}

std::string trim_ascii(const std::string& value) {
  const std::size_t begin = value.find_first_not_of(" \t");
  if (begin == std::string::npos) return {};
  const std::size_t end = value.find_last_not_of(" \t");
  return value.substr(begin, end - begin + 1);
}

bool parse_request_head(const std::string& bytes, std::size_t header_end,
                        Request* request) {
  const std::size_t line_end = bytes.find("\r\n");
  if (line_end == std::string::npos || line_end >= header_end) return false;
  const std::string line = bytes.substr(0, line_end);
  const std::size_t first_space = line.find(' ');
  const std::size_t second_space =
      first_space == std::string::npos ? std::string::npos
                                       : line.find(' ', first_space + 1);
  if (first_space == std::string::npos || second_space == std::string::npos ||
      line.find(' ', second_space + 1) != std::string::npos) {
    return false;
  }
  const std::string version = line.substr(second_space + 1);
  if (version != "HTTP/1.1" && version != "HTTP/1.0") return false;
  request->method = line.substr(0, first_space);
  request->target = line.substr(first_space + 1, second_space - first_space - 1);
  if (request->method.empty() || request->target.empty() ||
      request->target.front() != '/') {
    return false;
  }
  std::size_t offset = line_end + 2;
  while (offset < header_end) {
    const std::size_t end = bytes.find("\r\n", offset);
    if (end == std::string::npos || end > header_end) return false;
    if (end == offset) break;
    const std::string header = bytes.substr(offset, end - offset);
    const std::size_t colon = header.find(':');
    if (colon == std::string::npos || colon == 0) return false;
    const std::string name = lower_ascii(trim_ascii(header.substr(0, colon)));
    const std::string value = trim_ascii(header.substr(colon + 1));
    if (name.empty() || !request->headers.emplace(name, value).second) return false;
    offset = end + 2;
  }
  return true;
}

bool parse_content_length(const std::string& text,
                          std::size_t* length) noexcept {
  if (text.empty()) return false;
  errno = 0;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
  if (errno != 0 || end == nullptr || *end != '\0') return false;
  *length = static_cast<std::size_t>(parsed);
  return static_cast<unsigned long long>(*length) == parsed;
}

bool read_request(int socket, Request* request, int* error_status,
                  std::string* error_message) noexcept {
  std::string input;
  std::size_t delimiter = std::string::npos;
  while (input.size() <= kMaximumHeaderBytes) {
    delimiter = input.find("\r\n\r\n");
    if (delimiter != std::string::npos) break;
    char buffer[4096];
    const ssize_t count = recv(socket, buffer, sizeof(buffer), 0);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) break;
    input.append(buffer, static_cast<std::size_t>(count));
  }
  if (delimiter == std::string::npos || delimiter + 4 > kMaximumHeaderBytes ||
      !parse_request_head(input, delimiter + 2, request)) {
    *error_status = 400;
    *error_message = "Malformed HTTP request.";
    return false;
  }
  if (request->headers.find("transfer-encoding") != request->headers.end()) {
    *error_status = 400;
    *error_message = "Transfer-Encoding is unsupported; send Content-Length.";
    return false;
  }
  std::size_t body_bytes = 0;
  const auto content_length = request->headers.find("content-length");
  if (content_length != request->headers.end() &&
      !parse_content_length(content_length->second, &body_bytes)) {
    *error_status = 400;
    *error_message = "Content-Length is invalid.";
    return false;
  }
  if (request->method == "POST" && content_length == request->headers.end()) {
    *error_status = 411;
    *error_message = "POST requests require Content-Length.";
    return false;
  }
  if (body_bytes > kMaximumBodyBytes) {
    *error_status = 413;
    *error_message = "Request body exceeds 1048576 bytes.";
    return false;
  }
  const std::size_t body_begin = delimiter + 4;
  while (input.size() - body_begin < body_bytes) {
    char buffer[4096];
    const std::size_t missing = body_bytes - (input.size() - body_begin);
    const ssize_t count = recv(socket, buffer, std::min(sizeof(buffer), missing), 0);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) {
      *error_status = 400;
      *error_message = "Request body ended before Content-Length.";
      return false;
    }
    input.append(buffer, static_cast<std::size_t>(count));
  }
  request->body = input.substr(body_begin, body_bytes);
  return true;
}

std::string request_path(const Request& request) {
  const std::size_t query = request.target.find('?');
  return request.target.substr(0, query);
}

Response control_route(const Request& request, const RouteContext& context) {
  const std::string path = request_path(request);
  if (request.method != "GET") {
    return {405,
            error_body("Only GET is supported by this route.",
                       "method_not_allowed"),
            "Allow: GET\r\n"};
  }
  if (path == "/health") {
    const std::size_t queued = context.gate->waiting();
    const bool active = context.gate->active();
    return {200,
            std::string("{\"status\":\"ok\",\"ready\":true,\"model\":\"") +
                context.model_id + "\",\"single_flight\":true,\"sessions\":1," +
                "\"queue_active\":" + (active ? "true" : "false") +
                ",\"queue_depth\":" + std::to_string(queued) +
                ",\"cancelled_requests\":" +
                std::to_string(context.cancelled_requests->load(
                    std::memory_order_relaxed)) + "}",
            std::string()};
  }
  if (path == "/v1/models") {
    return {200,
            std::string("{\"object\":\"list\",\"data\":[{\"id\":\"") +
                context.model_id +
                "\",\"object\":\"model\",\"created\":0,\"owned_by\":"
                "\"quartz-watch-38\"}]}",
            std::string()};
  }
  return {404, error_body("Route not found.", "not_found"), std::string()};
}

std::string response_id() {
  const std::uint64_t value =
      g_response_counter.fetch_add(1, std::memory_order_relaxed);
  return "chatcmpl-qw38-" + std::to_string(value);
}

std::string responses_id() {
  const std::uint64_t value =
      g_response_counter.fetch_add(1, std::memory_order_relaxed);
  const std::int64_t micros = std::chrono::duration_cast<std::chrono::microseconds>(
                                  std::chrono::system_clock::now().time_since_epoch())
                                  .count();
  return "resp_qw38_" + std::to_string(micros) + "_" +
         std::to_string(value);
}

std::int64_t unix_seconds() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string queue_headers(const qw38::server::QueueTiming& timing) {
  return "X-QW38-Queue-Depth: " + std::to_string(timing.depth_on_arrival) +
         "\r\nX-QW38-Queue-Us: " + std::to_string(timing.microseconds) + "\r\n";
}

std::string usage_json(const qw38::server::GenerationResult& result) {
  return std::string("{\"prompt_tokens\":") +
         std::to_string(result.prompt_tokens) + ",\"completion_tokens\":" +
         std::to_string(result.completion_tokens) + ",\"total_tokens\":" +
         std::to_string(result.prompt_tokens + result.completion_tokens) + "}";
}

std::string responses_usage_json(
    const qw38::server::GenerationResult& result) {
  return std::string("{\"input_tokens\":") +
         std::to_string(result.prompt_tokens) +
         ",\"input_tokens_details\":{\"cached_tokens\":0},\"output_tokens\":" +
         std::to_string(result.completion_tokens) +
         ",\"output_tokens_details\":{\"reasoning_tokens\":0},\"total_tokens\":" +
         std::to_string(result.prompt_tokens + result.completion_tokens) + "}";
}

std::string responses_output_json(
    const std::string& id, const qw38::server::AssistantOutput& output) {
  std::string json = "[";
  bool comma = false;
  if (!output.reasoning.empty()) {
    json += "{\"id\":" + qw38::server::quote_json("rs_" + id.substr(5)) +
            ",\"type\":\"reasoning\",\"summary\":[{\"type\":\"summary_text\",\"text\":" +
            qw38::server::quote_json(output.reasoning) + "}]}";
    comma = true;
  }
  if (!output.content.empty() || output.tool_calls.empty()) {
    if (comma) json += ',';
    json += "{\"id\":" + qw38::server::quote_json("msg_" + id.substr(5)) +
            ",\"type\":\"message\",\"status\":\"completed\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":" +
            qw38::server::quote_json(output.content) +
            ",\"annotations\":[]}]}";
    comma = true;
  }
  for (std::size_t index = 0; index < output.tool_calls.size(); ++index) {
    if (comma) json += ',';
    const std::string suffix = id.substr(5) + "_" + std::to_string(index);
    json += "{\"id\":" + qw38::server::quote_json("fc_" + suffix) +
            ",\"type\":\"function_call\",\"status\":\"completed\",\"call_id\":" +
            qw38::server::quote_json("call_" + suffix) + ",\"name\":" +
            qw38::server::quote_json(output.tool_calls[index].name) +
            ",\"arguments\":" +
            qw38::server::quote_json(output.tool_calls[index].arguments_json) +
            "}";
    comma = true;
  }
  return json + "]";
}

std::string responses_json(
    const std::string& id, std::int64_t created,
    const std::string& previous_response_id, bool store,
    const qw38::server::GenerationResult& result) {
  const bool incomplete = result.finish_reason == "length";
  return "{\"id\":" + qw38::server::quote_json(id) +
         ",\"object\":\"response\",\"created_at\":" +
         std::to_string(created) + ",\"status\":\"" +
         (incomplete ? "incomplete" : "completed") +
         "\",\"error\":null,\"incomplete_details\":" +
         (incomplete ? "{\"reason\":\"max_output_tokens\"}" : "null") +
         ",\"model\":\"" + kModelId + "\",\"output\":" +
         responses_output_json(id, result.output) +
         ",\"parallel_tool_calls\":false,\"previous_response_id\":" +
         (previous_response_id.empty()
              ? "null"
              : qw38::server::quote_json(previous_response_id)) +
         ",\"store\":" + (store ? "true" : "false") +
         ",\"usage\":" + responses_usage_json(result) + "}";
}

std::string tool_calls_json(const std::string& id,
                            const qw38::server::AssistantOutput& output) {
  std::string json = "[";
  for (std::size_t index = 0; index < output.tool_calls.size(); ++index) {
    if (index != 0) json += ',';
    json += "{\"id\":" + qw38::server::quote_json(
                "call_" + id.substr(9) + "_" + std::to_string(index));
    json += ",\"type\":\"function\",\"function\":{\"name\":" +
            qw38::server::quote_json(output.tool_calls[index].name) +
            ",\"arguments\":" +
            qw38::server::quote_json(output.tool_calls[index].arguments_json) +
            "}}";
  }
  return json + "]";
}

std::string stream_tool_calls_json(
    const std::string& id, const qw38::server::AssistantOutput& output) {
  std::string json = "[";
  for (std::size_t index = 0; index < output.tool_calls.size(); ++index) {
    if (index != 0) json += ',';
    json += "{\"index\":" + std::to_string(index) + ",\"id\":" +
            qw38::server::quote_json("call_" + id.substr(9) + "_" +
                                     std::to_string(index));
    json += ",\"type\":\"function\",\"function\":{\"name\":" +
            qw38::server::quote_json(output.tool_calls[index].name) +
            ",\"arguments\":" +
            qw38::server::quote_json(output.tool_calls[index].arguments_json) +
            "}}";
  }
  return json + "]";
}

std::string completion_json(const std::string& id, std::int64_t created,
                            const std::string& model_id,
                            const qw38::server::GenerationResult& result) {
  std::string message = "{\"role\":\"assistant\",\"content\":";
  message += result.output.tool_calls.empty()
                 ? qw38::server::quote_json(result.output.content)
                 : "null";
  if (!result.output.reasoning.empty()) {
    message += ",\"reasoning_content\":" +
               qw38::server::quote_json(result.output.reasoning);
  }
  if (!result.output.tool_calls.empty()) {
    message += ",\"tool_calls\":" + tool_calls_json(id, result.output);
  }
  message += "}";
  return "{\"id\":" + qw38::server::quote_json(id) +
         ",\"object\":\"chat.completion\",\"created\":" +
         std::to_string(created) + ",\"model\":\"" + model_id +
         "\",\"choices\":[{\"index\":0,\"message\":" + message +
         ",\"finish_reason\":" +
         qw38::server::quote_json(result.finish_reason) +
         "}],\"usage\":" + usage_json(result) +
         ",\"system_fingerprint\":\"qw38-sm120-v1\"}";
}

struct StreamContext final {
  int socket = -1;
  std::string id;
  std::int64_t created = 0;
  std::string model_id;
};

bool send_sse(int socket, const std::string& json) noexcept {
  return send_all(socket, "data: " + json + "\n\n");
}

std::string chunk_json(const StreamContext& context,
                       const std::string& delta,
                       const char* finish_reason) {
  return "{\"id\":" + qw38::server::quote_json(context.id) +
         ",\"object\":\"chat.completion.chunk\",\"created\":" +
         std::to_string(context.created) + ",\"model\":\"" + context.model_id +
         "\",\"choices\":[{\"index\":0,\"delta\":" + delta +
         ",\"finish_reason\":" +
         (finish_reason == nullptr ? "null"
                                   : qw38::server::quote_json(finish_reason)) +
         "}],\"system_fingerprint\":\"qw38-sm120-v1\"}";
}

bool stream_delta(void* opaque, qw38::server::DeltaKind kind,
                  const std::string& bytes) noexcept {
  auto* context = static_cast<StreamContext*>(opaque);
  const char* field = kind == qw38::server::DeltaKind::kReasoning
                          ? "reasoning_content"
                          : "content";
  const std::string delta = std::string("{\"") + field + "\":" +
                            qw38::server::quote_json(bytes) + "}";
  return send_sse(context->socket, chunk_json(*context, delta, nullptr));
}

struct ResponsesStreamContext final {
  int socket = -1;
  std::string id;
  std::uint64_t sequence = 0;
  std::size_t next_output_index = 0;
  std::size_t reasoning_index = 0;
  std::size_t message_index = 0;
  bool reasoning_started = false;
  bool message_started = false;
};

bool response_event(ResponsesStreamContext* context, const std::string& type,
                    const std::string& fields) noexcept {
  const std::string json = "{\"type\":" + qw38::server::quote_json(type) +
                           ",\"sequence_number\":" +
                           std::to_string(context->sequence++) +
                           (fields.empty() ? "" : "," + fields) + "}";
  return send_all(context->socket, "event: " + type + "\ndata: " + json +
                                       "\n\n");
}

bool responses_stream_delta(void* opaque, qw38::server::DeltaKind kind,
                            const std::string& bytes) noexcept {
  auto* context = static_cast<ResponsesStreamContext*>(opaque);
  if (kind == qw38::server::DeltaKind::kReasoning) {
    if (!context->reasoning_started) {
      context->reasoning_started = true;
      context->reasoning_index = context->next_output_index++;
      const std::string item =
          "{\"id\":" + qw38::server::quote_json("rs_" + context->id.substr(5)) +
          ",\"type\":\"reasoning\",\"summary\":[]}";
      if (!response_event(context, "response.output_item.added",
                          "\"output_index\":" +
                              std::to_string(context->reasoning_index) +
                              ",\"item\":" + item) ||
          !response_event(context, "response.reasoning_summary_part.added",
                          "\"item_id\":" +
                              qw38::server::quote_json("rs_" +
                                                       context->id.substr(5)) +
                              ",\"output_index\":" +
                              std::to_string(context->reasoning_index) +
                              ",\"summary_index\":0,\"part\":{\"type\":\"summary_text\",\"text\":\"\"}")) {
        return false;
      }
    }
    return response_event(
        context, "response.reasoning_summary_text.delta",
        "\"item_id\":" +
            qw38::server::quote_json("rs_" + context->id.substr(5)) +
            ",\"output_index\":" + std::to_string(context->reasoning_index) +
            ",\"summary_index\":0,\"delta\":" +
            qw38::server::quote_json(bytes));
  }
  if (!context->message_started) {
    context->message_started = true;
    context->message_index = context->next_output_index++;
    const std::string item =
        "{\"id\":" + qw38::server::quote_json("msg_" + context->id.substr(5)) +
        ",\"type\":\"message\",\"status\":\"in_progress\",\"role\":\"assistant\",\"content\":[]}";
    if (!response_event(context, "response.output_item.added",
                        "\"output_index\":" +
                            std::to_string(context->message_index) +
                            ",\"item\":" + item) ||
        !response_event(context, "response.content_part.added",
                        "\"item_id\":" +
                            qw38::server::quote_json("msg_" +
                                                     context->id.substr(5)) +
                            ",\"output_index\":" +
                            std::to_string(context->message_index) +
                            ",\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[]}")) {
      return false;
    }
  }
  return response_event(
      context, "response.output_text.delta",
      "\"item_id\":" +
          qw38::server::quote_json("msg_" + context->id.substr(5)) +
          ",\"output_index\":" + std::to_string(context->message_index) +
          ",\"content_index\":0,\"delta\":" +
          qw38::server::quote_json(bytes));
}

int status_for(const qw38::Status& status) noexcept {
  if (status.code() == qw38::StatusCode::kCancelled) return 499;
  if (status.code() == qw38::StatusCode::kInternal ||
      status.code() == qw38::StatusCode::kIoError) return 500;
  return 400;
}

void serve_chat(int socket, const Request& request,
                const RouteContext& context) noexcept {
  const auto content_type = request.headers.find("content-type");
  if (content_type == request.headers.end() ||
      lower_ascii(content_type->second).find("application/json") != 0) {
    write_response(socket,
                   {400,
                    error_body("Content-Type must be application/json.",
                               "invalid_content_type"),
                    std::string()});
    return;
  }
  qw38::server::Json root;
  qw38::Status status = qw38::server::parse_json(request.body, 64, &root);
  qw38::server::ChatRequest chat;
  if (status.is_ok()) {
    status = qw38::server::parse_chat_request(root, context.model_id.c_str(),
                                              &chat);
  }
  if (!status.is_ok()) {
    write_response(socket,
                   {400, error_body(status.message(),
                                    qw38::status_code_name(status.code())),
                    std::string()});
    return;
  }

  DisconnectMonitor monitor(socket);
  qw38::server::QueueTiming timing;
  status = context.gate->acquire(monitor.cancelled(), &timing);
  if (!status.is_ok()) {
    if (status.code() == qw38::StatusCode::kCancelled) {
      context.cancelled_requests->fetch_add(1, std::memory_order_relaxed);
    }
    monitor.stop();
    if (!monitor.cancelled()->load(std::memory_order_relaxed)) {
      write_response(socket,
                     {status_for(status),
                      error_body(status.message(),
                                 qw38::status_code_name(status.code())),
                      std::string()});
    }
    return;
  }

  const std::string id = response_id();
  const std::int64_t created = unix_seconds();
  StreamContext stream{socket, id, created, context.model_id};
  bool stream_started = false;
  if (chat.stream) {
    stream_started = write_headers(socket, 200, "text/event-stream",
                                   "Cache-Control: no-cache\r\n" +
                                       queue_headers(timing),
                                   nullptr);
    if (stream_started) {
      stream_started = send_sse(
          socket, chunk_json(stream, "{\"role\":\"assistant\",\"content\":\"\"}",
                             nullptr));
    }
    if (!stream_started) monitor.cancelled()->store(true, std::memory_order_relaxed);
  }

  qw38::server::GenerationResult result;
  if (status.is_ok() && (!chat.stream || stream_started)) {
    status = qw38::server::generate_chat(
        *context.engine, context.session, chat, context.end_token,
        monitor.cancelled(), chat.stream ? stream_delta : nullptr,
        chat.stream ? static_cast<void*>(&stream) : nullptr, &result);
  }
  const qw38::Status release_status = context.gate->release();
  if (status.is_ok() && !release_status.is_ok()) status = release_status;
  if (status.code() == qw38::StatusCode::kCancelled) {
    context.cancelled_requests->fetch_add(1, std::memory_order_relaxed);
  }

  if (chat.stream && stream_started) {
    if (status.is_ok()) {
      if (!result.output.tool_calls.empty()) {
        const std::string delta = "{\"tool_calls\":" +
                                  stream_tool_calls_json(id, result.output) + "}";
        send_sse(socket, chunk_json(stream, delta, nullptr));
      }
      send_sse(socket, chunk_json(stream, "{}", result.finish_reason.c_str()));
      if (chat.include_usage) {
        const std::string usage =
            "{\"id\":" + qw38::server::quote_json(id) +
            ",\"object\":\"chat.completion.chunk\",\"created\":" +
            std::to_string(created) + ",\"model\":\"" + context.model_id +
            "\",\"choices\":[],\"usage\":" + usage_json(result) + "}";
        send_sse(socket, usage);
      }
    } else if (!monitor.cancelled()->load(std::memory_order_relaxed)) {
      send_sse(socket,
               error_body(status.message(),
                          qw38::status_code_name(status.code())));
    }
    send_all(socket, "data: [DONE]\n\n");
  } else if (status.is_ok()) {
    const std::string body =
        completion_json(id, created, context.model_id, result);
    write_response(socket, {200, body, queue_headers(timing)});
  } else if (!monitor.cancelled()->load(std::memory_order_relaxed)) {
    write_response(socket,
                   {status_for(status),
                    error_body(status.message(),
                               qw38::status_code_name(status.code())),
                    queue_headers(timing)});
  }
  monitor.stop();
}

bool same_tools(const std::vector<qw38::server::Json>& left,
                const std::vector<qw38::server::Json>& right) {
  if (left.size() != right.size()) return false;
  for (std::size_t index = 0; index < left.size(); ++index) {
    if (qw38::server::dump_json(left[index], true) !=
        qw38::server::dump_json(right[index], true)) {
      return false;
    }
  }
  return true;
}

qw38::Status prepare_continuation(
    const RouteContext& context,
    qw38::server::ResponsesRequest* response) noexcept {
  if (response->previous_response_id.empty()) return qw38::Status::ok();
  qw38::server::ResponseRecord previous;
  qw38::Status status =
      context.response_store->load(response->previous_response_id, &previous);
  if (!status.is_ok()) return status;
  auto& generation = response->generation;
  if (!generation.tool_schemas.empty() &&
      !same_tools(generation.tool_schemas, previous.tool_schemas)) {
    return {qw38::StatusCode::kInvalidArgument,
            "tools must be omitted or exactly match the stored response"};
  }
  if (generation.tool_schemas.empty()) {
    generation.tool_schemas = previous.tool_schemas;
    generation.chat.canonical_tools.clear();
    for (const qw38::server::Json& tool : previous.tool_schemas) {
      generation.chat.canonical_tools.push_back(
          qw38::server::dump_json(tool, true));
    }
  }
  if (generation.tool_choice != qw38::server::ToolChoice::kAuto) {
    return {qw38::StatusCode::kUnimplemented,
            "continued requests support tool_choice=auto only"};
  }
  std::vector<qw38::Token> suffix;
  status = context.engine->render_followup(
      generation.messages, generation.chat.enable_thinking, &suffix);
  if (!status.is_ok()) return status;
  generation.prepared_prompt = std::move(previous.tokens);
  generation.prepared_prompt.insert(generation.prepared_prompt.end(),
                                    suffix.begin(), suffix.end());
  return qw38::Status::ok();
}

bool finish_responses_stream(
    ResponsesStreamContext* stream,
    const qw38::server::GenerationResult& result) noexcept {
  if (stream->reasoning_started) {
    const std::string base =
        "\"item_id\":" +
        qw38::server::quote_json("rs_" + stream->id.substr(5)) +
        ",\"output_index\":" + std::to_string(stream->reasoning_index) +
        ",\"summary_index\":0";
    if (!response_event(stream, "response.reasoning_summary_text.done",
                        base + ",\"text\":" +
                                   qw38::server::quote_json(
                                       result.output.reasoning)) ||
        !response_event(
            stream, "response.reasoning_summary_part.done",
            base + ",\"part\":{\"type\":\"summary_text\",\"text\":" +
                qw38::server::quote_json(result.output.reasoning) + "}")) {
      return false;
    }
    const std::string item =
        "{\"id\":" +
        qw38::server::quote_json("rs_" + stream->id.substr(5)) +
        ",\"type\":\"reasoning\",\"summary\":[{\"type\":\"summary_text\",\"text\":" +
        qw38::server::quote_json(result.output.reasoning) + "}]}";
    if (!response_event(stream, "response.output_item.done",
                        "\"output_index\":" +
                            std::to_string(stream->reasoning_index) +
                            ",\"item\":" + item)) {
      return false;
    }
  }
  if (stream->message_started) {
    const std::string base =
        "\"item_id\":" +
        qw38::server::quote_json("msg_" + stream->id.substr(5)) +
        ",\"output_index\":" + std::to_string(stream->message_index) +
        ",\"content_index\":0";
    if (!response_event(stream, "response.output_text.done",
                        base + ",\"text\":" +
                                   qw38::server::quote_json(
                                       result.output.content)) ||
        !response_event(
            stream, "response.content_part.done",
            base + ",\"part\":{\"type\":\"output_text\",\"text\":" +
                qw38::server::quote_json(result.output.content) +
                ",\"annotations\":[]}")) {
      return false;
    }
    const std::string item =
        "{\"id\":" +
        qw38::server::quote_json("msg_" + stream->id.substr(5)) +
        ",\"type\":\"message\",\"status\":\"completed\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":" +
        qw38::server::quote_json(result.output.content) +
        ",\"annotations\":[]}]}";
    if (!response_event(stream, "response.output_item.done",
                        "\"output_index\":" +
                            std::to_string(stream->message_index) +
                            ",\"item\":" + item)) {
      return false;
    }
  }
  for (std::size_t index = 0; index < result.output.tool_calls.size(); ++index) {
    const std::size_t output_index = stream->next_output_index++;
    const std::string suffix = stream->id.substr(5) + "_" + std::to_string(index);
    const auto& call = result.output.tool_calls[index];
    const std::string pending =
        "{\"id\":" + qw38::server::quote_json("fc_" + suffix) +
        ",\"type\":\"function_call\",\"status\":\"in_progress\",\"call_id\":" +
        qw38::server::quote_json("call_" + suffix) + ",\"name\":" +
        qw38::server::quote_json(call.name) + ",\"arguments\":\"\"}";
    const std::string base =
        "\"item_id\":" + qw38::server::quote_json("fc_" + suffix) +
        ",\"output_index\":" + std::to_string(output_index);
    if (!response_event(stream, "response.output_item.added",
                        "\"output_index\":" + std::to_string(output_index) +
                            ",\"item\":" + pending) ||
        !response_event(stream, "response.function_call_arguments.delta",
                        base + ",\"delta\":" +
                                   qw38::server::quote_json(
                                       call.arguments_json)) ||
        !response_event(stream, "response.function_call_arguments.done",
                        base + ",\"arguments\":" +
                                   qw38::server::quote_json(
                                       call.arguments_json))) {
      return false;
    }
    const std::string done =
        "{\"id\":" + qw38::server::quote_json("fc_" + suffix) +
        ",\"type\":\"function_call\",\"status\":\"completed\",\"call_id\":" +
        qw38::server::quote_json("call_" + suffix) + ",\"name\":" +
        qw38::server::quote_json(call.name) + ",\"arguments\":" +
        qw38::server::quote_json(call.arguments_json) + "}";
    if (!response_event(stream, "response.output_item.done",
                        "\"output_index\":" + std::to_string(output_index) +
                            ",\"item\":" + done)) {
      return false;
    }
  }
  return true;
}

void serve_responses(int socket, const Request& request,
                     const RouteContext& context) noexcept {
  const auto content_type = request.headers.find("content-type");
  if (content_type == request.headers.end() ||
      lower_ascii(content_type->second).find("application/json") != 0) {
    write_response(socket,
                   {400,
                    error_body("Content-Type must be application/json.",
                               "invalid_content_type"),
                    std::string()});
    return;
  }
  qw38::server::Json root;
  qw38::Status status = qw38::server::parse_json(request.body, 64, &root);
  qw38::server::ResponsesRequest response;
  if (status.is_ok()) {
    status = qw38::server::parse_responses_request(root, &response);
  }
  if (!status.is_ok()) {
    write_response(socket,
                   {400, error_body(status.message(),
                                    qw38::status_code_name(status.code())),
                    std::string()});
    return;
  }

  DisconnectMonitor monitor(socket);
  qw38::server::QueueTiming timing;
  status = context.gate->acquire(monitor.cancelled(), &timing);
  const bool acquired = status.is_ok();
  if (status.is_ok()) status = prepare_continuation(context, &response);
  if (!status.is_ok()) {
    if (acquired) context.gate->release();
    if (status.code() == qw38::StatusCode::kCancelled) {
      context.cancelled_requests->fetch_add(1, std::memory_order_relaxed);
    }
    monitor.stop();
    if (!monitor.cancelled()->load(std::memory_order_relaxed)) {
      write_response(socket,
                     {status_for(status),
                      error_body(status.message(),
                                 qw38::status_code_name(status.code())),
                      queue_headers(timing)});
    }
    return;
  }

  const std::string id = responses_id();
  const std::int64_t created = unix_seconds();
  ResponsesStreamContext stream{socket, id};
  bool stream_started = false;
  if (response.generation.stream) {
    stream_started = write_headers(socket, 200, "text/event-stream",
                                   "Cache-Control: no-cache\r\n" +
                                       queue_headers(timing),
                                   nullptr);
    const std::string pending =
        "{\"id\":" + qw38::server::quote_json(id) +
        ",\"object\":\"response\",\"created_at\":" +
        std::to_string(created) +
        ",\"status\":\"in_progress\",\"error\":null,\"incomplete_details\":null,\"model\":\"" +
        kModelId + "\",\"output\":[],\"parallel_tool_calls\":false,\"previous_response_id\":" +
        (response.previous_response_id.empty()
             ? "null"
             : qw38::server::quote_json(response.previous_response_id)) +
        ",\"store\":" + (response.store ? "true" : "false") + "}";
    if (stream_started) {
      stream_started = response_event(&stream, "response.created",
                                      "\"response\":" + pending) &&
                       response_event(&stream, "response.in_progress",
                                      "\"response\":" + pending);
    }
    if (!stream_started) monitor.cancelled()->store(true, std::memory_order_relaxed);
  }

  qw38::server::GenerationResult result;
  if (!response.generation.stream || stream_started) {
    status = qw38::server::generate_chat(
        *context.engine, context.session, response.generation, context.end_token,
        monitor.cancelled(),
        response.generation.stream ? responses_stream_delta : nullptr,
        response.generation.stream ? static_cast<void*>(&stream) : nullptr,
        &result);
  }
  if (status.is_ok() && response.store) {
    qw38::server::ResponseRecord record;
    status = context.session->tokens(&record.tokens);
    record.tool_schemas = response.generation.tool_schemas;
    if (status.is_ok()) status = context.response_store->save(id, record);
  }
  const qw38::Status release_status = context.gate->release();
  if (status.is_ok() && !release_status.is_ok()) status = release_status;
  if (status.code() == qw38::StatusCode::kCancelled) {
    context.cancelled_requests->fetch_add(1, std::memory_order_relaxed);
  }

  const std::string completed = responses_json(
      id, created, response.previous_response_id, response.store, result);
  if (response.generation.stream && stream_started) {
    if (status.is_ok()) {
      if (finish_responses_stream(&stream, result)) {
        response_event(&stream, result.finish_reason == "length"
                                    ? "response.incomplete"
                                    : "response.completed",
                       "\"response\":" + completed);
      }
    } else if (!monitor.cancelled()->load(std::memory_order_relaxed)) {
      response_event(&stream, "error",
                     "\"error\":{\"message\":" +
                         qw38::server::quote_json(status.message()) +
                         ",\"type\":\"invalid_request_error\",\"code\":" +
                         qw38::server::quote_json(
                             qw38::status_code_name(status.code())) +
                         "}");
    }
  } else if (status.is_ok()) {
    write_response(socket, {200, completed, queue_headers(timing)});
  } else if (!monitor.cancelled()->load(std::memory_order_relaxed)) {
    write_response(socket,
                   {status_for(status),
                    error_body(status.message(),
                               qw38::status_code_name(status.code())),
                    queue_headers(timing)});
  }
  monitor.stop();
}

void handle_connection(int socket, const RouteContext& context,
                       ConnectionTracker* tracker) noexcept {
  ConnectionOwner owner(socket, tracker);
  timeval timeout{5, 0};
  setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  Request request;
  int error_status = 400;
  std::string error_message;
  if (!read_request(socket, &request, &error_status, &error_message)) {
    write_response(socket,
                   {error_status, error_body(error_message, "bad_request"),
                    std::string()});
    return;
  }
  const std::string path = request_path(request);
  if (path == "/v1/chat/completions") {
    if (request.method != "POST") {
      write_response(socket,
                     {405,
                      error_body("Chat Completions requires POST.",
                                 "method_not_allowed"),
                      "Allow: POST\r\n"});
    } else {
      serve_chat(socket, request, context);
    }
    return;
  }
  if (path == "/v1/responses") {
    if (request.method != "POST") {
      write_response(socket,
                     {405,
                      error_body("Responses requires POST.",
                                 "method_not_allowed"),
                      "Allow: POST\r\n"});
    } else {
      serve_responses(socket, request, context);
    }
    return;
  }
  write_response(socket, control_route(request, context));
}

int open_listener(const Options& options, std::uint16_t* bound_port) {
  const int listener = socket(AF_INET, SOCK_STREAM, 0);
  if (listener < 0) return -1;
  int reuse = 1;
  setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(options.port);
  if (inet_pton(AF_INET, options.host.c_str(), &address.sin_addr) != 1 ||
      bind(listener, reinterpret_cast<const sockaddr*>(&address),
           sizeof(address)) != 0) {
    close(listener);
    return -1;
  }
  socklen_t size = sizeof(address);
  if (getsockname(listener, reinterpret_cast<sockaddr*>(&address), &size) != 0) {
    close(listener);
    return -1;
  }
  *bound_port = ntohs(address.sin_port);
  return listener;
}

int report(const qw38::Status& status) {
  std::cerr << qw38::status_code_name(status.code()) << ": "
            << status.message() << '\n';
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!parse_options(argc, argv, &options)) {
    usage(std::cerr);
    return 2;
  }
  std::cout << kBrand << '\n';

  std::uint16_t bound_port = 0;
  const int listener = open_listener(options, &bound_port);
  if (listener < 0) {
    std::cerr << "io_error: cannot bind HTTP listener: " << std::strerror(errno)
              << '\n';
    return 1;
  }

  qw38::Engine engine;
  qw38::server::ResponseStore response_store(options.response_dir);
  qw38::Status status = response_store.open();
  if (status.is_ok()) status = qw38::Engine::open(options.model, &engine);
  std::string model_id = kModelId;
  if (status.is_ok()) status = engine.admitted_model(&model_id);
  std::unique_ptr<qw38::Session> session;
  if (status.is_ok()) status = engine.create_session(&session);
  std::vector<qw38::Token> end_tokens;
  if (status.is_ok()) status = engine.encode("<|im_end|>", &end_tokens);
  if (status.is_ok() && end_tokens.size() != 1) {
    status = {qw38::StatusCode::kIncompatibleArtifact,
              "chat terminator is not one token"};
  }
  if (!status.is_ok()) {
    close(listener);
    return report(status);
  }
  if (listen(listener, 64) != 0) {
    std::cerr << "io_error: cannot listen on HTTP socket: "
              << std::strerror(errno) << '\n';
    close(listener);
    return 1;
  }

  qw38::server::SingleFlightGate gate;
  std::atomic<std::uint64_t> cancelled_requests{0};
  RouteContext context{&engine, session.get(), end_tokens[0], &gate,
                       &cancelled_requests, model_id, &response_store};
  ConnectionTracker tracker;
  struct sigaction action {};
  action.sa_handler = stop_server;
  sigemptyset(&action.sa_mask);
  sigaction(SIGINT, &action, nullptr);
  sigaction(SIGTERM, &action, nullptr);
  g_listener = listener;
  std::cout << "listening=http://" << options.host << ':' << bound_port
            << " model=" << model_id << " sessions=1 response_dir="
            << response_store.directory() << "\n"
            << std::flush;

  while (!g_stop) {
    const int client = accept(listener, nullptr, nullptr);
    if (client < 0) {
      if (errno == EINTR) continue;
      if (g_stop || errno == EBADF || errno == EINVAL) break;
      continue;
    }
    tracker.start();
    std::thread(handle_connection, client, std::cref(context), &tracker).detach();
  }
  gate.shutdown();
  if (g_listener >= 0) {
    close(listener);
    g_listener = -1;
  }
  tracker.wait();
  return 0;
}
