#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr const char* kVariants[] = {"o2_fmad_false", "o3_fmad_false",
                                     "o3_fmad_true"};
constexpr int kVariantCount = 3;
constexpr const char* kFamily = "q4_prompt_mmq";
constexpr double kPromptLayers = 64.0;
constexpr double kComponentGateMs = 5.0;

const char* json_bool(bool value) { return value ? "true" : "false"; }

std::string variant_bin(const char* variant) {
  return std::string("build/cuda/opt068/") + variant +
         "/qw38-cuda-opt068-codegen-test";
}

std::string variant_obj(const char* variant) {
  return std::string("build/cuda/opt068/") + variant + "/q4_prompt_mmq.cuda.o";
}

std::string variant_stamp(const char* variant) {
  return std::string("build/cuda/opt068/") + variant + "/nvccflags.stamp";
}

std::string slurp(const char* path) {
  FILE* file = std::fopen(path, "r");
  if (file == nullptr) return "";
  std::string text;
  char buf[512];
  while (std::fgets(buf, sizeof(buf), file) != nullptr) text += buf;
  std::fclose(file);
  while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
    text.pop_back();
  }
  return text;
}

std::string sha256sum(const char* path) {
  const std::string cmd = std::string("sha256sum ") + path;
  FILE* pipe = popen(cmd.c_str(), "r");
  if (pipe == nullptr) return "unavailable";
  char line[256];
  std::string out;
  if (std::fgets(line, sizeof(line), pipe) != nullptr) out = line;
  pclose(pipe);
  const auto space = out.find(' ');
  if (space != std::string::npos) out.resize(space);
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
    out.pop_back();
  }
  return out.empty() ? "unavailable" : out;
}

bool parse_ms(const std::string& line, const char* variant, double* ms) {
  const std::string needle = std::string("complete_ffn_ms variant=") + variant;
  if (line.find(needle) == std::string::npos) return false;
  const auto pos = line.find(" ms=");
  if (pos == std::string::npos) return false;
  *ms = std::strtod(line.c_str() + pos + 4, nullptr);
  return true;
}

int run_child(const char* variant, int argc, char** argv, double* mean_ms,
              bool* have_mean) {
  const std::string bin = variant_bin(variant);
  std::vector<char*> child_argv;
  child_argv.push_back(const_cast<char*>(bin.c_str()));
  for (int index = 1; index < argc; ++index) child_argv.push_back(argv[index]);
  child_argv.push_back(nullptr);
  int fds[2];
  if (pipe(fds) != 0) {
    std::fprintf(stderr, "pipe failed for %s\n", variant);
    return 1;
  }
  const pid_t pid = fork();
  if (pid < 0) {
    std::fprintf(stderr, "fork failed for %s\n", variant);
    return 1;
  }
  if (pid == 0) {
    close(fds[0]);
    if (dup2(fds[1], STDOUT_FILENO) < 0) _exit(127);
    close(fds[1]);
    execv(bin.c_str(), child_argv.data());
    std::fprintf(stderr, "execv failed for %s\n", bin.c_str());
    _exit(127);
  }
  close(fds[1]);
  FILE* out = fdopen(fds[0], "r");
  if (out == nullptr) {
    close(fds[0]);
    return 1;
  }
  char line[1024];
  while (std::fgets(line, sizeof(line), out) != nullptr) {
    std::fputs(line, stdout);
    double parsed = 0.0;
    if (parse_ms(line, variant, &parsed)) {
      *mean_ms = parsed;
      *have_mean = true;
    }
  }
  std::fclose(out);
  int status = 0;
  if (waitpid(pid, &status, 0) < 0) return 1;
  if (WIFEXITED(status)) return WEXITSTATUS(status);
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  std::printf("opt068_family=%s chosen_from=OPT-061_prompt-ffn "
              "control_flags=-O2 --fmad=false host_flags_unchanged=true "
              "strict_objects_unchanged=true use_fast_math=false\n",
              kFamily);
  if (access("build/q4_prompt_mmq.cuda.o", R_OK) == 0) {
    std::printf("opt068_production_object=build/q4_prompt_mmq.cuda.o "
                "sha256=%s flags=NVCCFLAGS\n",
                sha256sum("build/q4_prompt_mmq.cuda.o").c_str());
  } else {
    std::printf("opt068_production_object=build/q4_prompt_mmq.cuda.o "
                "sha256=not_built_this_target flags=NVCCFLAGS\n");
  }
  for (int index = 0; index < kVariantCount; ++index) {
    const char* variant = kVariants[index];
    const std::string stamp = variant_stamp(variant);
    const std::string object = variant_obj(variant);
    std::printf("opt068_variant id=%s stamp=%s flags=%s object=%s sha256=%s\n",
                variant, stamp.c_str(), slurp(stamp.c_str()).c_str(),
                object.c_str(), sha256sum(object.c_str()).c_str());
  }

  int rc = 0;
  double means[kVariantCount] = {0.0, 0.0, 0.0};
  bool have_ms[kVariantCount] = {false, false, false};
  std::string hashes[kVariantCount];
  for (int index = 0; index < kVariantCount; ++index) {
    hashes[index] = sha256sum(variant_obj(kVariants[index]).c_str());
  }
  for (int index = 0; index < kVariantCount; ++index) {
    const char* variant = kVariants[index];
    std::printf("opt068_run variant=%s\n", variant);
    const int child =
        run_child(variant, argc, argv, &means[index], &have_ms[index]);
    if (child != 0) {
      std::fprintf(stderr, "variant %s failed rc=%d\n", variant, child);
      rc = child;
    }
  }

  FILE* make_file = std::fopen("Makefile", "r");
  bool host_o2 = false;
  bool host_contract_off = false;
  bool nvcc_o2 = false;
  bool nvcc_fmad_false = false;
  bool fast_math = false;
  if (make_file != nullptr) {
    char line[1024];
    while (std::fgets(line, sizeof(line), make_file) != nullptr) {
      if (std::strstr(line, "CXXFLAGS :=") != nullptr &&
          std::strstr(line, "-O2") != nullptr) {
        host_o2 = true;
      }
      if (std::strstr(line, "CXXFLAGS :=") != nullptr &&
          std::strstr(line, "-ffp-contract=off") != nullptr) {
        host_contract_off = true;
      }
      if (std::strstr(line, "NVCCFLAGS :=") != nullptr &&
          std::strstr(line, "-O2") != nullptr) {
        nvcc_o2 = true;
      }
      if (std::strstr(line, "NVCCFLAGS :=") != nullptr &&
          std::strstr(line, "--fmad=false") != nullptr) {
        nvcc_fmad_false = true;
      }
      if (std::strstr(line, "use_fast_math") != nullptr) fast_math = true;
    }
    std::fclose(make_file);
  }
  std::printf(
      "strict_isolation host_o2=%s host_fp_contract_off=%s nvcc_o2=%s "
      "nvcc_fmad_false=%s use_fast_math=%s\n",
      json_bool(host_o2), json_bool(host_contract_off), json_bool(nvcc_o2),
      json_bool(nvcc_fmad_false), json_bool(fast_math));
  if (!host_o2 || !host_contract_off || !nvcc_o2 || !nvcc_fmad_false ||
      fast_math) {
    rc = 1;
  }

  const char* workload = nullptr;
  for (int index = 1; index + 1 < argc; ++index) {
    if (std::strcmp(argv[index], "--workload") == 0) workload = argv[index + 1];
  }
  bool keep_o3 = false;
  bool keep_fma = false;
  const char* winner = "o2_fmad_false";
  if (workload != nullptr && (std::strcmp(workload, "screen") == 0 ||
                              std::strcmp(workload, "acceptance") == 0) &&
      have_ms[0] && have_ms[1] && have_ms[2] && rc == 0) {
    const bool o3_code_diff = hashes[0] != hashes[1];
    const bool fma_code_diff = hashes[0] != hashes[2];
    const double o3_save = (means[0] - means[1]) * kPromptLayers;
    const double fma_save = (means[0] - means[2]) * kPromptLayers;
    keep_o3 = o3_code_diff && means[1] < means[0] && o3_save >= kComponentGateMs;
    keep_fma =
        fma_code_diff && means[2] < means[0] && fma_save >= kComponentGateMs;
    if (keep_fma && (!keep_o3 || means[2] < means[1])) {
      winner = "o3_fmad_true";
    } else if (keep_o3) {
      winner = "o3_fmad_false";
    }
    std::printf(
        "screen_compare o2_ms=%.9g o3_ms=%.9g fma_ms=%.9g "
        "o3_save_prompt_ms=%.9g fma_save_prompt_ms=%.9g o3_code_diff=%s "
        "fma_code_diff=%s keep_o3=%s keep_fma=%s winner=%s installed=%s\n",
        means[0], means[1], means[2], o3_save, fma_save,
        json_bool(o3_code_diff), json_bool(fma_code_diff), json_bool(keep_o3),
        json_bool(keep_fma), winner, json_bool(keep_o3 || keep_fma));
  } else if (workload != nullptr && (std::strcmp(workload, "screen") == 0 ||
                                     std::strcmp(workload, "acceptance") == 0)) {
    std::printf("screen_compare incomplete rc=%d have_ms=%s,%s,%s\n", rc,
                json_bool(have_ms[0]), json_bool(have_ms[1]),
                json_bool(have_ms[2]));
  }
  std::printf(
      "keep_rule complete_cost_and_v2_quality cosmetic_o3_rejected=true "
      "component_gate_ms=%.1f prompt_layers=%.0f winner=%s installed=%s\n",
      kComponentGateMs, kPromptLayers, winner, json_bool(keep_o3 || keep_fma));

  std::printf(
      "QW38_OPT068_CODEGEN_RESULT={\"schema_version\":1,\"task\":\"OPT-068\","
      "\"family\":\"%s\",\"variants\":3,\"winner\":\"%s\",\"installed\":%s,"
      "\"keep_o3\":%s,\"keep_fma\":%s,\"claims_throughput\":false}\n",
      kFamily, winner, json_bool(keep_o3 || keep_fma), json_bool(keep_o3),
      json_bool(keep_fma));
  std::printf("status=%s\n", rc == 0 ? "passed" : "failed");
  return rc;
}
