#include "compiler/compile.hpp"
#include "compiler/error.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

namespace {

void usage() {
  std::cerr
      << "Usage: qw38-compile --checkpoint DIR --output FILE [--no-verify]\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path checkpoint;
  std::filesystem::path output;
  qw38::compiler::CompileOptions options{};
  for (int i = 1; i < argc; ++i) {
    std::string_view arg{argv[i]};
    if (arg == "--checkpoint" && i + 1 < argc) {
      checkpoint = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      output = argv[++i];
    } else if (arg == "--no-verify") {
      options.verify_reconstruction = false;
    } else if (arg == "--help" || arg == "-h") {
      usage();
      return 0;
    } else {
      std::cerr << "unknown argument: " << arg << '\n';
      usage();
      return 2;
    }
  }
  if (checkpoint.empty() || output.empty()) {
    usage();
    return 2;
  }
  auto result = qw38::compiler::compile_identity(checkpoint, output, options);
  if (!result) {
    std::cerr << qw38::compiler::error_message(result.error()) << '\n';
    return 1;
  }
  std::cout << "wrote " << result->identity.path.string()
            << " bytes=" << result->identity.size_bytes
            << " language_instances=" << result->language_instances
            << " mtp_instances=" << result->mtp_instances
            << " included_tensors=" << result->included_tensors
            << " vision_excluded=" << result->vision_excluded
            << " peak_rss_bytes=" << result->peak_rss_bytes << '\n';
  return 0;
}
