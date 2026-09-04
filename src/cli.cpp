#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "qw38/engine.h"

namespace {

constexpr const char* kBrand =
    "Quartz Watch 38 — Swiss-made inference for Qwen3.8-27B. It ticks fast.";
constexpr std::uint32_t kVocabularySize = 248320;

struct Options final {
  std::string model;
  std::string system;
  std::string prompt;
  std::string load;
  std::string save;
  std::vector<std::string> stops;
  qw38::ChatOptions chat;
  qw38::SamplerConfig sampler;
  std::size_t max_tokens = 256;
  std::size_t context = 0;
};

void usage(std::ostream& output) {
  output
      << "usage: qw38 MODEL [options]\n"
      << "  --prompt TEXT          run one turn instead of interactive input\n"
      << "  --system TEXT          initial system instruction\n"
      << "  --reasoning MODE       off, low, medium, or xhigh\n"
      << "  --max-tokens N         generated token limit (default 256)\n"
      << "  --ctx N                host context tokens (default 4096, max 8192)\n"
      << "  --temperature F        0 selects greedy decoding (default 0)\n"
      << "  --top-p F              nucleus probability in (0,1] (default 1)\n"
      << "  --top-k N              candidate limit; 0 means unlimited\n"
      << "  --seed N               deterministic sampler seed\n"
      << "  --stop TEXT            additional stop text; repeatable\n"
      << "  --load PATH            restore a session checkpoint\n"
      << "  --save PATH            save after every completed turn\n"
      << "interactive commands: /help, /save PATH, /load PATH, /reset, /quit\n";
}

bool parse_size(const char* text, std::size_t* value) {
  if (text == nullptr || *text == '\0') return false;
  errno = 0;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(text, &end, 10);
  if (errno != 0 || *end != '\0') return false;
  *value = static_cast<std::size_t>(parsed);
  return static_cast<unsigned long long>(*value) == parsed;
}

bool parse_u64(const char* text, std::uint64_t* value) {
  std::size_t parsed = 0;
  if (!parse_size(text, &parsed)) return false;
  *value = static_cast<std::uint64_t>(parsed);
  return static_cast<std::size_t>(*value) == parsed;
}

bool parse_float(const char* text, float* value) {
  if (text == nullptr || *text == '\0') return false;
  errno = 0;
  char* end = nullptr;
  const float parsed = std::strtof(text, &end);
  if (errno != 0 || *end != '\0') return false;
  *value = parsed;
  return true;
}

bool parse_options(int argc, char** argv, Options* options) {
  if (argc < 2) return false;
  if (argc == 2 && std::string(argv[1]) == "--help") {
    usage(std::cout);
    std::exit(0);
  }
  options->model = argv[1];
  options->sampler.temperature = 0.0F;
  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help") {
      usage(std::cout);
      std::exit(0);
    }
    if (index + 1 >= argc) return false;
    const char* value = argv[++index];
    if (argument == "--prompt") options->prompt = value;
    else if (argument == "--system") options->system = value;
    else if (argument == "--load") options->load = value;
    else if (argument == "--save") options->save = value;
    else if (argument == "--stop") options->stops.emplace_back(value);
    else if (argument == "--max-tokens") {
      if (!parse_size(value, &options->max_tokens) || options->max_tokens == 0) {
        return false;
      }
    } else if (argument == "--ctx") {
      if (!parse_size(value, &options->context) || options->context == 0 ||
          options->context > 8192) {
        return false;
      }
    } else if (argument == "--temperature") {
      if (!parse_float(value, &options->sampler.temperature)) return false;
    } else if (argument == "--top-p") {
      if (!parse_float(value, &options->sampler.top_p)) return false;
    } else if (argument == "--top-k") {
      std::size_t parsed = 0;
      if (!parse_size(value, &parsed) || parsed > kVocabularySize) return false;
      options->sampler.top_k = static_cast<std::uint32_t>(parsed);
    } else if (argument == "--seed") {
      if (!parse_u64(value, &options->sampler.seed)) return false;
    } else if (argument == "--reasoning") {
      if (std::string(value) == "off") {
        options->chat.enable_thinking = false;
      } else if (std::string(value) == "low" ||
                 std::string(value) == "medium" ||
                 std::string(value) == "xhigh") {
        options->chat.enable_thinking = true;
        options->chat.reasoning_effort = value;
      } else {
        return false;
      }
    } else {
      return false;
    }
  }
  return std::isfinite(options->sampler.temperature) &&
         options->sampler.temperature >= 0.0F &&
         std::isfinite(options->sampler.top_p) &&
         options->sampler.top_p > 0.0F && options->sampler.top_p <= 1.0F;
}

int report(const qw38::Status& status) {
  std::cerr << qw38::status_code_name(status.code()) << ": "
            << status.message() << '\n';
  return 1;
}

std::string trim(const std::string& value) {
  const std::size_t begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) return {};
  const std::size_t end = value.find_last_not_of(" \t\r\n");
  return value.substr(begin, end - begin + 1);
}

bool stopped_by_text(const std::string& text,
                     const std::vector<std::string>& stops,
                     std::size_t* visible_bytes) {
  for (const std::string& stop : stops) {
    if (!stop.empty() && text.size() >= stop.size() &&
        text.compare(text.size() - stop.size(), stop.size(), stop) == 0) {
      *visible_bytes = text.size() - stop.size();
      return true;
    }
  }
  *visible_bytes = text.size();
  return false;
}

void print_response(const std::string& decoded, bool thinking) {
  if (!thinking) {
    std::cout << "assistant> " << trim(decoded) << "\n";
    return;
  }
  const std::size_t end = decoded.find("</think>");
  if (end == std::string::npos) {
    std::cout << "reasoning> " << trim(decoded) << "\n";
    return;
  }
  const std::string reasoning = trim(decoded.substr(0, end));
  const std::string answer = trim(decoded.substr(end + 8));
  if (!reasoning.empty()) std::cout << "reasoning> " << reasoning << "\n";
  std::cout << "assistant> " << answer << "\n";
}

qw38::Status append_user(const qw38::Engine& engine,
                         const Options& options, const std::string& user,
                         bool first_turn,
                         std::vector<qw38::Token>* history) {
  std::vector<qw38::Token> suffix;
  qw38::Status status;
  if (first_turn) {
    std::vector<qw38::ChatMessage> messages;
    if (!options.system.empty()) {
      messages.push_back(
          {qw38::ChatRole::kSystem, options.system, std::string(), {}});
    }
    messages.push_back({qw38::ChatRole::kUser, user, std::string(), {}});
    status = engine.render_chat(messages, options.chat, &suffix);
  } else {
    status = engine.render_user_turn(user, options.chat.enable_thinking,
                                     &suffix);
  }
  if (status.is_ok()) {
    history->insert(history->end(), suffix.begin(), suffix.end());
  }
  return status;
}

qw38::Status generate(const qw38::Engine& engine, qw38::Session* session,
                      const Options& options, qw38::Token end_token,
                      std::vector<qw38::Token>* history) {
  qw38::Status status = session->sync(*history);
  std::vector<qw38::Token> generated;
  std::string decoded;
  bool closed = false;
  for (std::size_t index = 0;
       status.is_ok() && index < options.max_tokens; ++index) {
    qw38::Token token = 0;
    status = session->sample(options.sampler, &token);
    if (!status.is_ok()) break;
    status = session->eval(token);
    if (!status.is_ok()) break;
    history->push_back(token);
    if (token == end_token) {
      closed = true;
      break;
    }
    generated.push_back(token);
    status = engine.decode(generated, true, &decoded);
    std::size_t visible = decoded.size();
    if (status.is_ok() && stopped_by_text(decoded, options.stops, &visible)) {
      decoded.resize(visible);
      break;
    }
  }
  if (status.is_ok() && !closed) {
    status = session->eval(end_token);
    if (status.is_ok()) history->push_back(end_token);
  }
  if (!status.is_ok()) return status;
  print_response(decoded, options.chat.enable_thinking);
  if (!options.save.empty()) status = session->save(options.save);
  return status;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!parse_options(argc, argv, &options)) {
    usage(std::cerr);
    return 2;
  }
  std::cout << kBrand << '\n';
  qw38::Engine engine;
  qw38::Status status =
      qw38::Engine::open(options.model, options.context, &engine);
  if (!status.is_ok()) return report(status);
  std::unique_ptr<qw38::Session> session;
  status = engine.create_session(&session);
  if (!status.is_ok()) return report(status);
  std::vector<qw38::Token> end_tokens;
  status = engine.encode("<|im_end|>", &end_tokens);
  if (!status.is_ok()) return report(status);
  if (end_tokens.size() != 1) {
    std::cerr << "incompatible_artifact: chat terminator is not one token\n";
    return 1;
  }
  std::vector<qw38::Token> history;
  if (!options.load.empty()) {
    status = session->restore(options.load);
    if (status.is_ok()) status = session->tokens(&history);
    if (!status.is_ok()) return report(status);
  }

  if (!options.prompt.empty()) {
    status = append_user(engine, options, options.prompt, history.empty(),
                         &history);
    if (status.is_ok()) {
      status = generate(engine, session.get(), options, end_tokens[0], &history);
    }
    return status.is_ok() ? 0 : report(status);
  }

  std::cout << "Type /help for commands.\n";
  std::string line;
  while (std::cout << "user> " && std::getline(std::cin, line)) {
    if (line == "/quit") break;
    if (line == "/help") {
      usage(std::cout);
      continue;
    }
    if (line == "/reset") {
      status = session->sync({});
      if (status.is_ok()) history.clear();
      if (!status.is_ok()) std::cerr << status.message() << '\n';
      continue;
    }
    if (line.rfind("/save ", 0) == 0) {
      const std::string path = trim(line.substr(6));
      status = session->save(path);
      if (!status.is_ok()) std::cerr << status.message() << '\n';
      else std::cout << "saved> " << path << '\n';
      continue;
    }
    if (line.rfind("/load ", 0) == 0) {
      const std::string path = trim(line.substr(6));
      status = session->restore(path);
      if (status.is_ok()) status = session->tokens(&history);
      if (!status.is_ok()) std::cerr << status.message() << '\n';
      else std::cout << "restored> " << path << '\n';
      continue;
    }
    if (trim(line).empty()) continue;
    status = append_user(engine, options, line, history.empty(), &history);
    if (status.is_ok()) {
      status = generate(engine, session.get(), options, end_tokens[0], &history);
    }
    if (!status.is_ok()) std::cerr << status.message() << '\n';
  }
  return 0;
}
