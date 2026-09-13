#include "opt113_sitting_selectors.cuh"

#include <cstdio>
#include <cstring>

namespace {

constexpr char kPrefix[] = "QW38_OPT113_COUPLED_STACK_GATE_RESULT=";

const char* json_bool(bool value) { return value ? "true" : "false"; }

}  // namespace

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  const bool selected_ok = qw38::cuda::opt113_selected_pins_ok();
  const bool rejected_ok = qw38::cuda::opt113_rejected_not_leaked();

  bool control_apply = true;
  control_apply = control_apply && qw38::cuda::apply_q4_decode_ident(
                                       qw38::cuda::kOpt113ControlQ4, 4U);
  control_apply = control_apply && qw38::cuda::apply_attention_pipeline_ident(
                                       qw38::cuda::kOpt113ControlAttention);
  control_apply =
      control_apply && qw38::cuda::apply_decode_attention_crossover_threshold(
                           qw38::cuda::kOpt113ControlCrossover);
  const bool control_effective =
      std::strcmp(qw38::cuda::effective_q4_decode_path(),
                  qw38::cuda::kOpt113ControlQ4) == 0 &&
      std::strcmp(qw38::cuda::current_attention_pipeline_path(),
                  qw38::cuda::kOpt113ControlAttention) == 0 &&
      qw38::cuda::effective_decode_attention_crossover_threshold() ==
          qw38::cuda::kOpt113ControlCrossover;

  bool selected_apply = true;
  selected_apply = selected_apply && qw38::cuda::apply_q4_decode_ident(
                                         qw38::cuda::kOpt113SelectedQ4, 4U);
  selected_apply = selected_apply && qw38::cuda::apply_attention_pipeline_ident(
                                         qw38::cuda::kOpt113SelectedAttention);
  selected_apply =
      selected_apply && qw38::cuda::apply_decode_attention_crossover_threshold(
                            qw38::cuda::kOpt113SelectedCrossover);
  const bool selected_effective =
      std::strcmp(qw38::cuda::effective_q4_decode_path(),
                  qw38::cuda::kOpt113SelectedQ4) == 0 &&
      std::strcmp(qw38::cuda::current_attention_pipeline_path(),
                  qw38::cuda::kOpt113SelectedAttention) == 0 &&
      qw38::cuda::effective_decode_attention_crossover_threshold() ==
          qw38::cuda::kOpt113SelectedCrossover;

  const bool opt112_omitted = true;
  const bool pass = selected_ok && rejected_ok && control_apply &&
                    control_effective && selected_apply && selected_effective &&
                    opt112_omitted;
  std::printf(
      "%s{\"schema_version\":1,\"task\":\"OPT-113\","
      "\"selected_q4\":\"%s\",\"selected_attention\":\"%s\","
      "\"selected_crossover\":%d,\"vec128_pin\":\"%s\",\"gdn_pin\":\"%s\","
      "\"selected_pins_ok\":%s,\"rejected_not_leaked\":%s,"
      "\"control_apply\":%s,\"control_effective\":%s,"
      "\"selected_apply\":%s,\"selected_effective\":%s,"
      "\"opt112_deferred_omitted\":%s,\"pass\":%s}\n",
      kPrefix, qw38::cuda::selected_q4_decode_path(),
      qw38::cuda::selected_attention_pipeline_path(),
      qw38::cuda::selected_decode_attention_crossover_threshold(),
      qw38::cuda::selected_decode_attention_vec128_path(),
      qw38::cuda::selected_gdn_decode_path(), json_bool(selected_ok),
      json_bool(rejected_ok), json_bool(control_apply),
      json_bool(control_effective), json_bool(selected_apply),
      json_bool(selected_effective), json_bool(opt112_omitted),
      json_bool(pass));
  std::printf("status=%s\n", pass ? "passed" : "failed");
  return pass ? 0 : 1;
}
