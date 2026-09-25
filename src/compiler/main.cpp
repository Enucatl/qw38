#include "compiler/compile.hpp"
#include "compiler/error.hpp"

#include <filesystem>
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

namespace {

void usage() {
  std::cerr << "Usage: qw38-compile --checkpoint DIR --output FILE "
               "[--format identity|production|candidate|candidate-q4k] "
               "[--verify-reconstruction|--verify-only]\n";
}

std::string hex(qw38::format::Hash256 const& hash) {
  std::ostringstream out;
  for (auto byte : hash.bytes) {
    out << std::hex << std::setw(2) << std::setfill('0')
        << static_cast<unsigned int>(byte);
  }
  return out.str();
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path checkpoint;
  std::filesystem::path output;
  qw38::compiler::CompileOptions options{};
  bool verify_only = false;
  for (int i = 1; i < argc; ++i) {
    std::string_view arg{argv[i]};
    if (arg == "--checkpoint" && i + 1 < argc) {
      checkpoint = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      output = argv[++i];
    } else if (arg == "--verify-reconstruction") {
      options.verify_reconstruction = true;
    } else if (arg == "--verify-only") {
      verify_only = true;
    } else if (arg == "--format" && i + 1 < argc) {
      std::string_view mode{argv[++i]};
      if (mode == "identity") {
        options.format_policy = qw38::compiler::WeightFormatPolicy::IdentityBf16;
        options.revision.ident = qw38::compiler::kCompilerIdent;
      } else if (mode == "production") {
        options.format_policy = qw38::compiler::WeightFormatPolicy::ProductionV0;
        options.revision.ident = qw38::compiler::kProductionCompilerIdent;
      } else if (mode == "candidate") {
        options.format_policy = qw38::compiler::WeightFormatPolicy::CandidateV1;
        options.revision.ident = qw38::compiler::kCandidateCompilerIdent;
      } else if (mode == "candidate-q4k") {
        options.format_policy = qw38::compiler::WeightFormatPolicy::CandidateV2;
        options.revision.ident = qw38::compiler::kQ4KCandidateCompilerIdent;
      } else {
        std::cerr << "unknown format: " << mode << '\n';
        usage();
        return 2;
      }
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
  if (verify_only) {
    auto const started = std::chrono::steady_clock::now();
    auto verified = qw38::compiler::verify_compiled_artifact(
        output, checkpoint, options.format_policy, options.revision);
    if (!verified) {
      std::cerr << qw38::compiler::error_message(verified.error()) << '\n';
      return 1;
    }
    auto const elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    std::cout << "verified " << output.string()
              << " verify_ms=" << elapsed_ms
              << " peak_rss_bytes="
              << qw38::compiler::current_peak_rss_bytes()
              << " policy="
              << (options.format_policy == qw38::compiler::WeightFormatPolicy::CandidateV2
                      ? "candidate-q4k" : options.format_policy ==
                          qw38::compiler::WeightFormatPolicy::CandidateV1
                      ? "candidate"
                      : options.format_policy ==
                                qw38::compiler::WeightFormatPolicy::ProductionV0
                            ? "production"
                            : "identity")
              << '\n';
    return 0;
  }
  auto const started = std::chrono::steady_clock::now();
  auto result = qw38::compiler::compile_checkpoint(checkpoint, output, options);
  if (!result) {
    std::cerr << qw38::compiler::error_message(result.error()) << '\n';
    return 1;
  }
  std::cout << "wrote " << result->identity.path.string()
            << " compile_ms="
            << std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - started).count()
            << " bytes=" << result->identity.size_bytes
            << " language_instances=" << result->language_instances
            << " mtp_instances=" << result->mtp_instances
            << " included_tensors=" << result->included_tensors
            << " vision_excluded=" << result->vision_excluded
            << " peak_rss_bytes=" << result->peak_rss_bytes
            << " reconstruction_verified="
            << (result->reconstruction_verified ? "true" : "false")
            << " artifact_manifest=" << hex(result->identity.manifest_digest)
            << " source_metadata=" << hex(result->source_metadata_hash)
            << " config=" << hex(result->config_hash)
            << " tokenizer=" << hex(result->tokenizer_hash)
            << " compiler=" << result->identity.compiler.ident << ':'
            << result->identity.compiler.major << '.'
            << result->identity.compiler.minor << '.'
            << result->identity.compiler.patch
            << " policy="
            << (result->format_policy == qw38::compiler::WeightFormatPolicy::CandidateV2
                    ? "candidate-q4k" : result->format_policy ==
                        qw38::compiler::WeightFormatPolicy::IdentityBf16
                    ? "identity"
                    : result->format_policy ==
                              qw38::compiler::WeightFormatPolicy::CandidateV1
                          ? "candidate"
                          : "production")
            << '\n';
  return 0;
}
