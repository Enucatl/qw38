#include "response_store.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace qw38::server {
namespace {

constexpr const char* kSchema = "qw38.responses-prefix.v1";
constexpr const char* kModel = "qwen3.8-27b-q4_k_m";
constexpr std::size_t kMaximumRecordBytes = 16 * 1024 * 1024;

Status io_error(const std::string& action) noexcept {
  return {StatusCode::kIoError, action + ": " + std::strerror(errno)};
}

bool valid_id(const std::string& id) noexcept {
  if (id.size() < 12 || id.size() > 128 || id.rfind("resp_qw38_", 0) != 0) {
    return false;
  }
  for (char byte : id) {
    const bool valid = (byte >= 'a' && byte <= 'z') ||
                       (byte >= 'A' && byte <= 'Z') ||
                       (byte >= '0' && byte <= '9') || byte == '_' || byte == '-';
    if (!valid) return false;
  }
  return true;
}

Status make_directories(const std::string& path) noexcept {
  if (path.empty()) {
    return {StatusCode::kInvalidArgument, "response directory cannot be empty"};
  }
  std::string current;
  std::size_t offset = 0;
  if (path.front() == '/') {
    current = "/";
    offset = 1;
  }
  while (offset <= path.size()) {
    const std::size_t slash = path.find('/', offset);
    const std::string part = path.substr(
        offset, slash == std::string::npos ? std::string::npos : slash - offset);
    if (!part.empty()) {
      if (!current.empty() && current.back() != '/') current += '/';
      current += part;
      if (mkdir(current.c_str(), 0700) != 0 && errno != EEXIST) {
        return io_error("cannot create response directory " + current);
      }
      struct stat info {};
      if (stat(current.c_str(), &info) != 0 || !S_ISDIR(info.st_mode)) {
        return {StatusCode::kIoError, current + " is not a directory"};
      }
    }
    if (slash == std::string::npos) break;
    offset = slash + 1;
  }
  return Status::ok();
}

Status write_all(int descriptor, const std::string& bytes) noexcept {
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const ssize_t count =
        write(descriptor, bytes.data() + offset, bytes.size() - offset);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) return io_error("cannot write response record");
    offset += static_cast<std::size_t>(count);
  }
  return Status::ok();
}

}  // namespace

ResponseStore::ResponseStore(std::string directory) noexcept
    : directory_(std::move(directory)) {}

Status ResponseStore::open() noexcept { return make_directories(directory_); }

const std::string& ResponseStore::directory() const noexcept { return directory_; }

Status ResponseStore::save(const std::string& id,
                           const ResponseRecord& record) noexcept {
  if (!valid_id(id)) {
    return {StatusCode::kInvalidArgument, "invalid response id"};
  }
  std::vector<Json> tokens;
  tokens.reserve(record.tokens.size());
  for (Token token : record.tokens) {
    tokens.push_back(Json::number(std::to_string(token)));
  }
  Json root = Json::object_value(
      {{"model", Json::string(kModel)},
       {"schema", Json::string(kSchema)},
       {"tokens", Json::array_value(std::move(tokens))},
       {"tools", Json::array_value(record.tool_schemas)}});
  const std::string bytes = dump_json(root) + "\n";
  if (bytes.size() > kMaximumRecordBytes) {
    return {StatusCode::kResourceExhausted,
            "response continuation record exceeds 16 MiB"};
  }
  const std::string target = directory_ + "/" + id + ".json";
  const std::string temporary = target + ".tmp-" + std::to_string(getpid());
  const int descriptor =
      ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (descriptor < 0) return io_error("cannot create response record");
  Status status = write_all(descriptor, bytes);
  if (status.is_ok() && fsync(descriptor) != 0) {
    status = io_error("cannot sync response record");
  }
  if (close(descriptor) != 0 && status.is_ok()) {
    status = io_error("cannot close response record");
  }
  if (!status.is_ok()) {
    unlink(temporary.c_str());
    return status;
  }
  if (rename(temporary.c_str(), target.c_str()) != 0) {
    const Status failure = io_error("cannot publish response record");
    unlink(temporary.c_str());
    return failure;
  }
  const int directory = ::open(directory_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (directory < 0) {
    const Status failure = io_error("cannot open response directory for sync");
    unlink(target.c_str());
    return failure;
  }
  if (fsync(directory) != 0) {
    const Status failure = io_error("cannot sync response directory");
    close(directory);
    unlink(target.c_str());
    return failure;
  }
  if (close(directory) != 0) return io_error("cannot close response directory");
  return Status::ok();
}

Status ResponseStore::load(const std::string& id,
                           ResponseRecord* record) const noexcept {
  if (record == nullptr || !valid_id(id)) {
    return {StatusCode::kInvalidArgument, "invalid previous_response_id"};
  }
  const std::string path = directory_ + "/" + id + ".json";
  const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (descriptor < 0) {
    if (errno == ENOENT) {
      return {StatusCode::kInvalidArgument,
              "previous_response_id was not found or was not stored"};
    }
    return io_error("cannot open response record");
  }
  struct stat info {};
  if (fstat(descriptor, &info) != 0 || !S_ISREG(info.st_mode) ||
      info.st_size <= 0 ||
      static_cast<std::uint64_t>(info.st_size) > kMaximumRecordBytes) {
    close(descriptor);
    return {StatusCode::kIncompatibleArtifact,
            "response record has an invalid file size or type"};
  }
  std::string bytes(static_cast<std::size_t>(info.st_size), '\0');
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const ssize_t count = read(descriptor, bytes.data() + offset,
                               bytes.size() - offset);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) {
      close(descriptor);
      return io_error("cannot read response record");
    }
    offset += static_cast<std::size_t>(count);
  }
  if (close(descriptor) != 0) return io_error("cannot close response record");
  Json root;
  Status status = parse_json(bytes, 64, &root);
  const Json* schema = status.is_ok() ? root.find("schema") : nullptr;
  const Json* model = status.is_ok() ? root.find("model") : nullptr;
  const Json* token_values = status.is_ok() ? root.find("tokens") : nullptr;
  const Json* tools = status.is_ok() ? root.find("tools") : nullptr;
  if (!status.is_ok() || root.kind != JsonKind::kObject || schema == nullptr ||
      schema->kind != JsonKind::kString || schema->text != kSchema ||
      model == nullptr || model->kind != JsonKind::kString ||
      model->text != kModel || token_values == nullptr ||
      token_values->kind != JsonKind::kArray || token_values->array.empty() ||
      tools == nullptr || tools->kind != JsonKind::kArray) {
    return {StatusCode::kIncompatibleArtifact,
            "response record schema or model is incompatible"};
  }
  ResponseRecord loaded;
  loaded.tokens.reserve(token_values->array.size());
  for (const Json& value : token_values->array) {
    if (value.kind != JsonKind::kNumber || value.text.empty() ||
        value.text.find_first_not_of("0123456789") != std::string::npos) {
      return {StatusCode::kIncompatibleArtifact,
              "response record contains an invalid token"};
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value.text.c_str(), &end, 10);
    if (errno != 0 || end == nullptr || *end != '\0' || parsed > 248319) {
      return {StatusCode::kIncompatibleArtifact,
              "response record token exceeds the model vocabulary"};
    }
    loaded.tokens.push_back(static_cast<Token>(parsed));
  }
  loaded.tool_schemas = tools->array;
  *record = std::move(loaded);
  return Status::ok();
}

}  // namespace qw38::server
