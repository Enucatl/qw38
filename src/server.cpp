#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "qw38/engine.h"
#include "server_core.h"

namespace {

constexpr const char* kBrand =
    "Quartz Watch 38 — Swiss-made inference for Qwen3.8-27B. It ticks fast.";
constexpr const char* kModelId = "qwen3.8-27b-q4_k_m";
constexpr std::size_t kMaximumHeaderBytes = 64 * 1024;

volatile std::sig_atomic_t g_stop = 0;
volatile std::sig_atomic_t g_listener = -1;

struct Options final {
  std::string model;
  std::string host = "127.0.0.1";
  std::uint16_t port = 8080;
};

struct Request final {
  std::string method;
  std::string target;
};

struct Response final {
  int status = 200;
  std::string body;
  std::string extra_headers;
};

struct RouteContext final {
  qw38::server::SingleFlightGate* gate = nullptr;
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

void usage(std::ostream& output) {
  output << "usage: qw38-server MODEL [--host IPV4] [--port PORT]\n"
         << "  --host IPV4   listen address (default 127.0.0.1)\n"
         << "  --port PORT   TCP port; 0 asks the OS to choose (default 8080)\n";
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
    else if (argument == "--port") {
      if (!parse_port(value, &options->port)) return false;
    } else {
      return false;
    }
  }
  return !options->host.empty();
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
  return "Internal Server Error";
}

std::string error_body(const char* message, const char* code) {
  return std::string("{\"error\":{\"message\":\"") + message +
         "\",\"type\":\"invalid_request_error\",\"code\":\"" + code +
         "\"}}";
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

void write_response(int socket, const Response& response) noexcept {
  std::string header = "HTTP/1.1 " + std::to_string(response.status) + " " +
                       reason(response.status) + "\r\n";
  header += "Content-Type: application/json\r\n";
  header += "Content-Length: " + std::to_string(response.body.size()) + "\r\n";
  header += "Connection: close\r\n";
  header += "Server: qw38\r\n";
  header += response.extra_headers;
  header += "\r\n";
  if (send_all(socket, header)) send_all(socket, response.body);
}

bool parse_request(const std::string& bytes, Request* request) {
  const std::size_t line_end = bytes.find("\r\n");
  if (line_end == std::string::npos) return false;
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
  while (offset < bytes.size()) {
    const std::size_t end = bytes.find("\r\n", offset);
    if (end == std::string::npos) return false;
    if (end == offset) return true;
    const std::string header = bytes.substr(offset, end - offset);
    const std::size_t colon = header.find(':');
    if (colon == std::string::npos || colon == 0) return false;
    offset = end + 2;
  }
  return false;
}

Response route(const Request& request, const RouteContext& context) {
  if (request.method != "GET") {
    return {405, error_body("Only GET is supported by this route.",
                            "method_not_allowed"),
            "Allow: GET\r\n"};
  }
  const std::size_t query = request.target.find('?');
  const std::string path = request.target.substr(0, query);
  if (path == "/health") {
    const std::size_t queued = context.gate->waiting();
    const bool active = context.gate->active();
    return {200,
            std::string("{\"status\":\"ok\",\"ready\":true,\"model\":\"") +
                kModelId + "\",\"single_flight\":true,\"sessions\":1," +
                "\"queue_active\":" + (active ? "true" : "false") +
                ",\"queue_depth\":" + std::to_string(queued) + "}",
            std::string()};
  }
  if (path == "/v1/models") {
    return {200,
            std::string("{\"object\":\"list\",\"data\":[{\"id\":\"") +
                kModelId +
                "\",\"object\":\"model\",\"created\":0,\"owned_by\":"
                "\"quartz-watch-38\"}]}",
            std::string()};
  }
  return {404, error_body("Route not found.", "not_found"), std::string()};
}

void handle_connection(int socket, const RouteContext& context,
                       ConnectionTracker* tracker) noexcept {
  timeval timeout{5, 0};
  setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  std::string input;
  bool complete = false;
  while (input.size() < kMaximumHeaderBytes) {
    char buffer[4096];
    const ssize_t count = recv(socket, buffer, sizeof(buffer), 0);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) break;
    input.append(buffer, static_cast<std::size_t>(count));
    if (input.size() > kMaximumHeaderBytes) break;
    if (input.find("\r\n\r\n") != std::string::npos) {
      complete = true;
      break;
    }
  }
  Request request;
  if (!complete || !parse_request(input, &request)) {
    write_response(socket,
                   {400, error_body("Malformed HTTP request.", "bad_request"),
                    std::string()});
  } else {
    write_response(socket, route(request, context));
  }
  close(socket);
  tracker->finish();
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
  qw38::Status status = qw38::Engine::open(options.model, &engine);
  std::unique_ptr<qw38::Session> session;
  if (status.is_ok()) status = engine.create_session(&session);
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
  RouteContext context{&gate};
  ConnectionTracker tracker;
  struct sigaction action {};
  action.sa_handler = stop_server;
  sigemptyset(&action.sa_mask);
  sigaction(SIGINT, &action, nullptr);
  sigaction(SIGTERM, &action, nullptr);
  g_listener = listener;
  std::cout << "listening=http://" << options.host << ':' << bound_port
            << " model=" << kModelId << " sessions=1\n"
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
