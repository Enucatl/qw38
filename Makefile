CXX ?= g++
CC ?= cc
NVCC ?= nvcc
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror -fno-exceptions -fno-rtti -ffp-contract=off -pthread
CFLAGS := -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror
NVCCFLAGS := -std=c++17 -O2 -arch=sm_120 --expt-relaxed-constexpr --fmad=false -Xcompiler=-Wall,-Wextra,-Werror,-fno-exceptions,-fno-rtti,-ffp-contract=off,-pthread
CPPFLAGS := -Iinclude -Isrc -Ithird_party/utf8proc
BUILD_DIR := build
CUDA_BUILD_DIR := $(BUILD_DIR)/cuda
CUDA_STRICT_DIR := $(CUDA_BUILD_DIR)/strict
CUDA_EXPERIMENTAL_DIR := $(CUDA_BUILD_DIR)/experimental
DIAGNOSTIC_DIR := $(BUILD_DIR)/diagnostic
LIB_SOURCES := src/status.cpp src/sha256.cpp src/model.cpp src/tokenizer.cpp src/template.cpp src/quant.cpp src/tensor.cpp src/conversion.cpp src/projection.cpp src/weights.cpp src/mixer.cpp src/scheduler.cpp src/scalar_runtime.cpp src/gdn.cpp src/attention.cpp src/engine.cpp
LIB_OBJECTS := $(LIB_SOURCES:src/%.cpp=$(BUILD_DIR)/%.o)
THIRD_PARTY_OBJECTS := $(BUILD_DIR)/utf8proc.o
BINARIES := $(BUILD_DIR)/qw38 $(BUILD_DIR)/qw38-server $(BUILD_DIR)/qw38-bench $(BUILD_DIR)/qw38-eval
HOST_DIAGNOSTICS := $(BUILD_DIR)/qw38-server-core-test $(BUILD_DIR)/qw38-server-api-test $(BUILD_DIR)/qw38-responses-api-test $(BUILD_DIR)/qw38-opt044-production-numerics
CUDA_IMAGE := qw38-cuda:13.0.2
QUANT_MMV_CUDA_OBJECTS := $(BUILD_DIR)/quant_mmv.cuda.o $(BUILD_DIR)/q4_prompt_mmq.cuda.o $(BUILD_DIR)/q4k_decode_dots.cuda.o $(BUILD_DIR)/q8_decode_dots.cuda.o $(BUILD_DIR)/q6k_decode_dots.cuda.o
NVCC_CUDA_DEPS = -MMD -MP -MF $(@:.o=.d) -MT $@
NVCC_BIN_DEPS = -MMD -MP -MF $@.d -MT $@
CUDA_STRICT_FLAG_TEXT := $(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Icuda
CUDA_TRACE_FLAG_TEXT := $(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda
CUDA_EXPERIMENTAL_FLAG_TEXT := $(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Icuda -DQW38_CUDA_EXPERIMENTAL
CUDA_STRICT_STAMP := $(CUDA_STRICT_DIR)/nvccflags.stamp
CUDA_TRACE_STAMP := $(CUDA_STRICT_DIR)/nvccflags.trace.stamp
CUDA_EXPERIMENTAL_STAMP := $(CUDA_EXPERIMENTAL_DIR)/nvccflags.stamp
SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS := $(BUILD_DIR)/full_scheduler.trace.cuda.o $(BUILD_DIR)/scheduler_primitives.cuda.o $(QUANT_MMV_CUDA_OBJECTS) $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/attention_decode.cuda.o
CUDA_RELEASE_OBJECTS := $(CUDA_BUILD_DIR)/engine.o $(BUILD_DIR)/eval.cuda.o $(BUILD_DIR)/bench.cuda.o $(QUANT_MMV_CUDA_OBJECTS) $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/attention_decode.cuda.o $(BUILD_DIR)/scheduler_primitives.cuda.o $(BUILD_DIR)/full_scheduler.cuda.o $(BUILD_DIR)/checkpoint.cuda.o
CUDA_TRACE_OBJECTS := $(CUDA_BUILD_DIR)/engine.trace.o $(BUILD_DIR)/eval.trace.cuda.o $(BUILD_DIR)/full_scheduler.trace.cuda.o $(BUILD_DIR)/checkpoint.trace.cuda.o

.PHONY: all clean test diagnostic cuda-image cuda-build cuda-native cuda-products cuda-opt057-diagnostics cuda-opt058-diagnostics cuda-opt059-diagnostics cuda-opt060-diagnostics cuda-opt061-diagnostics cuda-opt062-diagnostics cuda-opt063-diagnostics cuda-opt064-diagnostics cuda-opt065-diagnostics cuda-opt066-diagnostics cuda-opt067-diagnostics cuda-opt068-diagnostics cuda-opt069-diagnostics cuda-opt070-diagnostics cuda-opt071-diagnostics cuda-opt073-diagnostics cuda-opt074-diagnostics cuda-opt075-diagnostics cuda-opt076-diagnostics cuda-opt077-diagnostics cuda-opt078-diagnostics cuda-opt079-diagnostics cuda-opt080-diagnostics cuda-opt081-diagnostics cuda-opt082-diagnostics cuda-opt083-diagnostics cuda-opt084-diagnostics cuda-opt085-diagnostics cuda-opt086-diagnostics cuda-opt087-diagnostics cuda-opt088-diagnostics cuda-opt089-diagnostics cuda-opt090-diagnostics cuda-opt091-diagnostics cuda-opt092-diagnostics cuda-opt093-diagnostics cuda-opt094-diagnostics cuda-opt095-diagnostics cuda-opt096-diagnostics cuda-opt097-diagnostics cuda-opt098-diagnostics cuda-opt099-diagnostics cuda-opt100-diagnostics cuda-opt101-diagnostics cuda-opt102-diagnostics cuda-opt103-diagnostics cuda-opt105-diagnostics FORCE

all: $(BINARIES) $(HOST_DIAGNOSTICS)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(DIAGNOSTIC_DIR):
	mkdir -p $@

$(DIAGNOSTIC_DIR)/%.o: src/%.cpp | $(DIAGNOSTIC_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DQW38_DIAGNOSTIC_TRACE -MMD -MP -c $< -o $@

$(BUILD_DIR)/utf8proc.o: third_party/utf8proc/utf8proc.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Ithird_party/utf8proc -MMD -MP -c $< -o $@

$(BUILD_DIR)/qw38: $(LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(BUILD_DIR)/cli.o
	$(CXX) $(CXXFLAGS) $^ -o $@

SERVER_OBJECTS := $(BUILD_DIR)/server_core.o $(BUILD_DIR)/server_json.o $(BUILD_DIR)/server_api.o $(BUILD_DIR)/server_generation.o $(BUILD_DIR)/responses_api.o $(BUILD_DIR)/response_store.o

$(BUILD_DIR)/qw38-server: $(LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(SERVER_OBJECTS) $(BUILD_DIR)/server.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/qw38-server-core-test: $(BUILD_DIR)/status.o $(BUILD_DIR)/server_core.o $(BUILD_DIR)/server_core_test.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/qw38-server-api-test: $(BUILD_DIR)/status.o $(THIRD_PARTY_OBJECTS) $(BUILD_DIR)/server_json.o $(BUILD_DIR)/server_api.o $(BUILD_DIR)/server_api_test.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/qw38-responses-api-test: $(BUILD_DIR)/status.o $(THIRD_PARTY_OBJECTS) $(BUILD_DIR)/server_json.o $(BUILD_DIR)/server_api.o $(BUILD_DIR)/responses_api.o $(BUILD_DIR)/response_store.o $(BUILD_DIR)/responses_api_test.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/qw38-bench: $(LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(BUILD_DIR)/server_json.o $(BUILD_DIR)/bench.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/qw38-eval: $(LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(BUILD_DIR)/eval.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/qw38-opt044-production-numerics: src/opt044_production_numerics.cpp $(BUILD_DIR)/quant.o $(BUILD_DIR)/status.o $(BUILD_DIR)/sha256.o | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

DIAGNOSTIC_OBJECTS := $(LIB_SOURCES:src/%.cpp=$(DIAGNOSTIC_DIR)/%.o) $(DIAGNOSTIC_DIR)/diagnostic_trace.o $(DIAGNOSTIC_DIR)/eval.o
DIAGNOSTIC_LIB_OBJECTS := $(LIB_SOURCES:src/%.cpp=$(DIAGNOSTIC_DIR)/%.o) $(DIAGNOSTIC_DIR)/diagnostic_trace.o

diagnostic: $(BUILD_DIR)/qw38-eval-diagnostic

$(BUILD_DIR)/qw38-eval-diagnostic: $(DIAGNOSTIC_OBJECTS) $(THIRD_PARTY_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

test: all
	uv run pytest

cuda-image:
	docker build -f docker/cuda.Dockerfile -t $(CUDA_IMAGE) .

cuda-build: cuda-image
	docker run --rm --gpus all --user "$$(id -u):$$(id -g)" -v "$$(pwd):/workspace" $(CUDA_IMAGE) make clean all cuda-products cuda-native

$(CUDA_BUILD_DIR):
	mkdir -p $@

$(CUDA_STRICT_DIR):
	mkdir -p $@

$(CUDA_EXPERIMENTAL_DIR):
	mkdir -p $@

FORCE:

$(CUDA_STRICT_STAMP): FORCE | $(CUDA_STRICT_DIR)
	@printf '%s\n' '$(CUDA_STRICT_FLAG_TEXT)' > $@.tmp
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else mv -f $@.tmp $@; fi

$(CUDA_TRACE_STAMP): FORCE | $(CUDA_STRICT_DIR)
	@printf '%s\n' '$(CUDA_TRACE_FLAG_TEXT)' > $@.tmp
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else mv -f $@.tmp $@; fi

$(CUDA_EXPERIMENTAL_STAMP): FORCE | $(CUDA_EXPERIMENTAL_DIR)
	@printf '%s\n' '$(CUDA_EXPERIMENTAL_FLAG_TEXT)' > $@.tmp
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else mv -f $@.tmp $@; fi

$(CUDA_EXPERIMENTAL_DIR)/%.cuda.o: cuda/%.cu $(CUDA_EXPERIMENTAL_STAMP) | $(CUDA_EXPERIMENTAL_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Icuda -DQW38_CUDA_EXPERIMENTAL $(NVCC_CUDA_DEPS) -c $< -o $@

$(CUDA_BUILD_DIR)/engine.o: src/engine.cpp include/qw38/engine.h cuda/full_scheduler.h $(CUDA_STRICT_STAMP) | $(CUDA_BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_CUDA_RUNTIME -Icuda $(NVCC_CUDA_DEPS) -c $< -o $@

$(CUDA_BUILD_DIR)/engine.trace.o: src/engine.cpp include/qw38/engine.h cuda/full_scheduler.h $(CUDA_TRACE_STAMP) | $(CUDA_BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_CUDA_RUNTIME -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_CUDA_DEPS) -c $< -o $@

CUDA_ENGINE_OBJECTS := $(filter-out $(BUILD_DIR)/engine.o,$(LIB_OBJECTS)) $(CUDA_BUILD_DIR)/engine.o $(THIRD_PARTY_OBJECTS)

cuda-products: $(CUDA_BUILD_DIR)/qw38 $(CUDA_BUILD_DIR)/qw38-server $(CUDA_BUILD_DIR)/qw38-bench $(CUDA_BUILD_DIR)/qw38-eval $(CUDA_BUILD_DIR)/qw38-eval-diagnostic

$(BUILD_DIR)/eval.cuda.o: src/eval.cpp include/qw38/engine.h cuda/full_scheduler.h $(CUDA_STRICT_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_CUDA_RUNTIME -Icuda $(NVCC_CUDA_DEPS) -c $< -o $@

$(BUILD_DIR)/eval.trace.cuda.o: src/eval.cpp include/qw38/engine.h src/diagnostic_trace.h cuda/full_scheduler.h $(CUDA_TRACE_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_CUDA_RUNTIME -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_CUDA_DEPS) -c $< -o $@

$(CUDA_BUILD_DIR)/qw38-eval: $(CUDA_ENGINE_OBJECTS) $(BUILD_DIR)/eval.cuda.o $(BUILD_DIR)/checkpoint.cuda.o $(BUILD_DIR)/full_scheduler.cuda.o $(BUILD_DIR)/scheduler_primitives.cuda.o $(QUANT_MMV_CUDA_OBJECTS) $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/attention_decode.cuda.o | $(CUDA_BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

CUDA_DIAGNOSTIC_ENGINE_OBJECTS := $(filter-out $(DIAGNOSTIC_DIR)/engine.o,$(DIAGNOSTIC_LIB_OBJECTS)) $(CUDA_BUILD_DIR)/engine.trace.o $(THIRD_PARTY_OBJECTS)

$(CUDA_BUILD_DIR)/qw38-eval-diagnostic: $(CUDA_DIAGNOSTIC_ENGINE_OBJECTS) $(BUILD_DIR)/eval.trace.cuda.o $(BUILD_DIR)/checkpoint.trace.cuda.o $(BUILD_DIR)/full_scheduler.trace.cuda.o $(BUILD_DIR)/scheduler_primitives.cuda.o $(QUANT_MMV_CUDA_OBJECTS) $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/attention_decode.cuda.o | $(CUDA_BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(CUDA_BUILD_DIR)/qw38: $(CUDA_ENGINE_OBJECTS) $(BUILD_DIR)/cli.o $(BUILD_DIR)/checkpoint.cuda.o $(BUILD_DIR)/full_scheduler.cuda.o $(BUILD_DIR)/scheduler_primitives.cuda.o $(QUANT_MMV_CUDA_OBJECTS) $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/attention_decode.cuda.o | $(CUDA_BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(CUDA_BUILD_DIR)/qw38-server: $(CUDA_ENGINE_OBJECTS) $(SERVER_OBJECTS) $(BUILD_DIR)/server.o $(BUILD_DIR)/checkpoint.cuda.o $(BUILD_DIR)/full_scheduler.cuda.o $(BUILD_DIR)/scheduler_primitives.cuda.o $(QUANT_MMV_CUDA_OBJECTS) $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/attention_decode.cuda.o | $(CUDA_BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILD_DIR)/bench.cuda.o: src/bench.cpp include/qw38/engine.h $(CUDA_STRICT_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_CUDA_RUNTIME $(NVCC_CUDA_DEPS) -c $< -o $@

$(CUDA_BUILD_DIR)/qw38-bench: $(CUDA_ENGINE_OBJECTS) $(THIRD_PARTY_OBJECTS) $(BUILD_DIR)/server_json.o $(BUILD_DIR)/bench.cuda.o $(BUILD_DIR)/checkpoint.cuda.o $(BUILD_DIR)/full_scheduler.cuda.o $(BUILD_DIR)/scheduler_primitives.cuda.o $(QUANT_MMV_CUDA_OBJECTS) $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/attention_decode.cuda.o | $(CUDA_BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

cuda-native: $(BUILD_DIR)/qw38-cuda-probe $(BUILD_DIR)/qw38-cuda-quant-test $(BUILD_DIR)/qw38-cuda-dispatch-tuning-test $(BUILD_DIR)/qw38-cuda-gdn-test $(BUILD_DIR)/qw38-cuda-gdn-chunk-test $(BUILD_DIR)/qw38-cuda-attention-test $(BUILD_DIR)/qw38-cuda-attention-chunk-test $(BUILD_DIR)/qw38-cuda-scheduler-primitives-test $(BUILD_DIR)/qw38-cuda-full-scheduler-test $(BUILD_DIR)/qw38-cuda-prefix-sync-test $(BUILD_DIR)/qw38-cuda-prompt-scheduler-test $(BUILD_DIR)/qw38-cuda-atomic-eval-test $(BUILD_DIR)/qw38-cuda-checkpoint-test $(BUILD_DIR)/qw38-cuda-memory-fit-test $(BUILD_DIR)/qw38-cuda-timing-test $(BUILD_DIR)/qw38-cuda-fusion-test $(BUILD_DIR)/qw38-cuda-graph-test $(BUILD_DIR)/qw38-cuda-optimization-engine-probe $(BUILD_DIR)/qw38-cuda-opt058-quality-baseline-test

cuda-opt057-diagnostics: $(BUILD_DIR)/qw38-cuda-optimization-engine-probe $(BUILD_DIR)/qw38-cuda-decode-oracle-test $(BUILD_DIR)/qw38-cuda-prefill-4k-oracle-test

cuda-opt069-diagnostics: $(BUILD_DIR)/qw38-cuda-optimization-engine-probe $(BUILD_DIR)/qw38-cuda-decode-oracle-test $(BUILD_DIR)/qw38-cuda-prefill-4k-oracle-test $(BUILD_DIR)/qw38-cuda-prefill-2k-parity-test $(BUILD_DIR)/qw38-cuda-opt058-quality-baseline-test $(BUILD_DIR)/qw38-cuda-checkpoint-test $(BUILD_DIR)/qw38-cuda-memory-fit-test

cuda-opt080-diagnostics: $(BUILD_DIR)/qw38-cuda-optimization-engine-probe $(BUILD_DIR)/qw38-cuda-decode-oracle-test $(BUILD_DIR)/qw38-cuda-prefill-4k-oracle-test $(BUILD_DIR)/qw38-cuda-prefill-2k-parity-test $(BUILD_DIR)/qw38-cuda-opt058-quality-baseline-test $(BUILD_DIR)/qw38-cuda-checkpoint-test $(BUILD_DIR)/qw38-cuda-memory-fit-test

cuda-opt081-diagnostics:
	@echo "OPT-081 host-only kernel-parity policy; no CUDA diagnostic binary"

cuda-opt082-diagnostics: $(BUILD_DIR)/qw38-cuda-opt082-kernel-parity-test

cuda-opt083-diagnostics: $(BUILD_DIR)/qw38-cuda-opt058-quality-baseline-test

cuda-opt084-diagnostics: $(BUILD_DIR)/qw38-cuda-opt058-quality-baseline-test

cuda-opt085-diagnostics: $(BUILD_DIR)/qw38-cuda-opt082-kernel-parity-test $(BUILD_DIR)/qw38-cuda-component-replay $(BUILD_DIR)/qw38-cuda-optimization-engine-probe $(BUILD_DIR)/qw38-cuda-opt058-quality-baseline-test

cuda-opt086-diagnostics: $(BUILD_DIR)/qw38-cuda-opt082-kernel-parity-test $(BUILD_DIR)/qw38-cuda-component-replay $(BUILD_DIR)/qw38-cuda-optimization-engine-probe $(BUILD_DIR)/qw38-cuda-opt058-quality-baseline-test

cuda-opt087-diagnostics:
	@echo "OPT-087 host-only historical leftover inventory; no CUDA diagnostic binary"

cuda-opt088-diagnostics: $(BUILD_DIR)/qw38-cuda-optimization-engine-probe $(BUILD_DIR)/qw38-cuda-decode-oracle-test $(BUILD_DIR)/qw38-cuda-prefill-4k-oracle-test $(BUILD_DIR)/qw38-cuda-prefill-2k-parity-test $(BUILD_DIR)/qw38-cuda-opt058-quality-baseline-test $(BUILD_DIR)/qw38-cuda-checkpoint-test $(BUILD_DIR)/qw38-cuda-memory-fit-test

cuda-opt089-diagnostics: $(BUILD_DIR)/qw38-cuda-opt082-kernel-parity-test $(BUILD_DIR)/qw38-cuda-component-replay $(BUILD_DIR)/qw38-cuda-optimization-engine-probe $(BUILD_DIR)/qw38-cuda-opt058-quality-baseline-test

cuda-opt090-diagnostics: $(BUILD_DIR)/qw38-cuda-opt060-engine-attribution-test $(BUILD_DIR)/qw38-cuda-component-replay $(BUILD_DIR)/qw38-cuda-optimization-engine-probe

cuda-opt091-diagnostics: $(BUILD_DIR)/qw38-cuda-opt082-kernel-parity-test $(BUILD_DIR)/qw38-cuda-component-replay $(BUILD_DIR)/qw38-cuda-optimization-engine-probe $(BUILD_DIR)/qw38-cuda-opt058-quality-baseline-test

cuda-opt092-diagnostics: $(BUILD_DIR)/qw38-cuda-opt092-q8-grouped-test $(BUILD_DIR)/qw38-cuda-component-replay $(BUILD_DIR)/qw38-cuda-optimization-engine-probe $(BUILD_DIR)/qw38-cuda-opt058-quality-baseline-test

cuda-opt093-diagnostics: $(BUILD_DIR)/qw38-cuda-opt093-q4-factored-test $(BUILD_DIR)/qw38-cuda-component-replay $(BUILD_DIR)/qw38-cuda-optimization-engine-probe $(BUILD_DIR)/qw38-cuda-opt058-quality-baseline-test

cuda-opt094-diagnostics: $(BUILD_DIR)/qw38-cuda-opt077-gdn-decode-test $(BUILD_DIR)/qw38-cuda-component-replay $(BUILD_DIR)/qw38-cuda-optimization-engine-probe

cuda-opt095-diagnostics: $(BUILD_DIR)/qw38-cuda-opt095-attention-gqa-test $(BUILD_DIR)/qw38-cuda-component-replay $(BUILD_DIR)/qw38-cuda-optimization-engine-probe

cuda-opt096-diagnostics: $(BUILD_DIR)/qw38-cuda-opt096-decode-graphs-test $(BUILD_DIR)/qw38-cuda-optimization-engine-probe

cuda-opt097-diagnostics: $(BUILD_DIR)/qw38-cuda-opt097-mmq-wait-test $(BUILD_DIR)/qw38-cuda-component-replay $(BUILD_DIR)/qw38-cuda-optimization-engine-probe $(BUILD_DIR)/qw38-cuda-opt058-quality-baseline-test

cuda-opt098-diagnostics: $(BUILD_DIR)/qw38-cuda-optimization-engine-probe $(BUILD_DIR)/qw38-cuda-decode-oracle-test $(BUILD_DIR)/qw38-cuda-prefill-4k-oracle-test $(BUILD_DIR)/qw38-cuda-prefill-2k-parity-test $(BUILD_DIR)/qw38-cuda-opt058-quality-baseline-test $(BUILD_DIR)/qw38-cuda-checkpoint-test $(BUILD_DIR)/qw38-cuda-memory-fit-test

cuda-opt099-diagnostics: $(BUILD_DIR)/qw38-cuda-opt060-engine-attribution-test $(BUILD_DIR)/qw38-cuda-component-replay

cuda-opt100-diagnostics: $(BUILD_DIR)/qw38-cuda-opt100-q8-aligned-test $(BUILD_DIR)/qw38-cuda-component-replay $(BUILD_DIR)/qw38-cuda-optimization-engine-probe $(BUILD_DIR)/qw38-cuda-opt058-quality-baseline-test

cuda-opt101-diagnostics: $(BUILD_DIR)/qw38-cuda-opt101-gdn-transposed-test $(BUILD_DIR)/qw38-cuda-component-replay $(BUILD_DIR)/qw38-cuda-optimization-engine-probe $(BUILD_DIR)/qw38-cuda-opt058-quality-baseline-test

cuda-opt102-diagnostics: $(BUILD_DIR)/qw38-cuda-opt102-q4-repack-test $(BUILD_DIR)/qw38-cuda-component-replay $(BUILD_DIR)/qw38-cuda-optimization-engine-probe $(BUILD_DIR)/qw38-cuda-opt058-quality-baseline-test

cuda-opt103-diagnostics: $(BUILD_DIR)/qw38-cuda-opt103-vector-attention-test $(BUILD_DIR)/qw38-cuda-component-replay $(BUILD_DIR)/qw38-cuda-optimization-engine-probe

cuda-opt105-diagnostics: $(BUILD_DIR)/qw38-cuda-opt105-mmq-double-x-test $(BUILD_DIR)/qw38-cuda-component-replay $(BUILD_DIR)/qw38-cuda-optimization-engine-probe $(BUILD_DIR)/qw38-cuda-opt058-quality-baseline-test

cuda-opt073-diagnostics: $(BUILD_DIR)/qw38-cuda-opt058-quality-baseline-test

cuda-opt058-diagnostics: $(BUILD_DIR)/qw38-cuda-opt058-quality-baseline-test

cuda-opt059-diagnostics: $(BUILD_DIR)/qw38-cuda-opt059-numerics-test

cuda-opt074-diagnostics: $(BUILD_DIR)/qw38-cuda-opt059-numerics-test $(BUILD_DIR)/qw38-cuda-opt043-activation-capture-test

cuda-opt060-diagnostics: $(BUILD_DIR)/qw38-cuda-opt060-engine-attribution-test

cuda-opt061-diagnostics: $(BUILD_DIR)/qw38-cuda-component-replay

cuda-opt071-diagnostics: $(BUILD_DIR)/qw38-cuda-opt060-engine-attribution-test $(BUILD_DIR)/qw38-cuda-component-replay

cuda-opt070-diagnostics: $(BUILD_DIR)/qw38-cuda-component-replay $(BUILD_DIR)/qw38-cuda-optimization-engine-probe

cuda-opt075-diagnostics: $(BUILD_DIR)/qw38-cuda-component-replay $(BUILD_DIR)/qw38-cuda-optimization-engine-probe $(BUILD_DIR)/qw38-cuda-opt062-q4-admission-test $(BUILD_DIR)/qw38-cuda-opt063-integer-ffn-test

cuda-opt076-diagnostics: $(BUILD_DIR)/qw38-cuda-opt076-q4-reduction-test $(BUILD_DIR)/qw38-cuda-component-replay $(BUILD_DIR)/qw38-cuda-optimization-engine-probe

cuda-opt077-diagnostics: $(BUILD_DIR)/qw38-cuda-opt077-gdn-decode-test $(BUILD_DIR)/qw38-cuda-component-replay $(BUILD_DIR)/qw38-cuda-optimization-engine-probe

cuda-opt078-diagnostics: $(BUILD_DIR)/qw38-cuda-opt078-decode-attention-test $(BUILD_DIR)/qw38-cuda-component-replay $(BUILD_DIR)/qw38-cuda-optimization-engine-probe

cuda-opt079-diagnostics: $(BUILD_DIR)/qw38-cuda-opt079-attention-kv-operands-test $(BUILD_DIR)/qw38-cuda-optimization-engine-probe

cuda-opt062-diagnostics: $(BUILD_DIR)/qw38-cuda-opt062-q4-admission-test

cuda-opt063-diagnostics: $(BUILD_DIR)/qw38-cuda-opt063-integer-ffn-test

cuda-opt064-diagnostics: $(BUILD_DIR)/qw38-cuda-opt064-q8-rows-test

cuda-opt065-diagnostics: $(BUILD_DIR)/qw38-cuda-opt065-mmq-tiles-test

cuda-opt066-diagnostics: $(BUILD_DIR)/qw38-cuda-opt066-mmq-x-pipeline-test

cuda-opt067-diagnostics: $(BUILD_DIR)/qw38-cuda-opt067-prompt-pair-test

cuda-opt068-diagnostics: $(BUILD_DIR)/qw38-cuda-opt068-codegen-test

# OPT-068 scoped Q4 prompt-MMQ codegen. Production/strict objects keep
# NVCCFLAGS (-O2 --fmad=false). Variant dirs have unique stamps/objects.
OPT068_O2_DIR := $(CUDA_BUILD_DIR)/opt068/o2_fmad_false
OPT068_O3_DIR := $(CUDA_BUILD_DIR)/opt068/o3_fmad_false
OPT068_O3FMA_DIR := $(CUDA_BUILD_DIR)/opt068/o3_fmad_true
OPT068_NVCC_BASE := -std=c++17 -arch=sm_120 --expt-relaxed-constexpr -Xcompiler=-Wall,-Wextra,-Werror,-fno-exceptions,-fno-rtti,-ffp-contract=off,-pthread
OPT068_O2_NVCCFLAGS := $(OPT068_NVCC_BASE) -O2 --fmad=false -Xptxas=-v
OPT068_O3_NVCCFLAGS := $(OPT068_NVCC_BASE) -O3 --fmad=false -Xptxas=-v
OPT068_O3FMA_NVCCFLAGS := $(OPT068_NVCC_BASE) -O3 --fmad=true -Xptxas=-v
OPT068_O2_FLAG_TEXT := $(NVCC) $(OPT068_O2_NVCCFLAGS) -Icuda
OPT068_O3_FLAG_TEXT := $(NVCC) $(OPT068_O3_NVCCFLAGS) -Icuda
OPT068_O3FMA_FLAG_TEXT := $(NVCC) $(OPT068_O3FMA_NVCCFLAGS) -Icuda
OPT068_O2_STAMP := $(OPT068_O2_DIR)/nvccflags.stamp
OPT068_O3_STAMP := $(OPT068_O3_DIR)/nvccflags.stamp
OPT068_O3FMA_STAMP := $(OPT068_O3FMA_DIR)/nvccflags.stamp
OPT068_FAMILY_DEPS := cuda/q4_prompt_mmq.cu cuda/quant_mmv.h cuda/production_numerics.h cuda/quant_mmq_mma.cuh cuda/mma.cuh cuda/pdl_launch.cuh
OPT068_BASE_CUDA_OBJECTS := $(filter-out $(BUILD_DIR)/q4_prompt_mmq.cuda.o,$(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS))
OPT068_VARIANT_LINK := $(OPT068_BASE_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS)

$(OPT068_O2_DIR) $(OPT068_O3_DIR) $(OPT068_O3FMA_DIR):
	mkdir -p $@

$(OPT068_O2_STAMP): FORCE | $(OPT068_O2_DIR)
	@printf '%s\n' '$(OPT068_O2_FLAG_TEXT)' > $@.tmp
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else mv -f $@.tmp $@; fi

$(OPT068_O3_STAMP): FORCE | $(OPT068_O3_DIR)
	@printf '%s\n' '$(OPT068_O3_FLAG_TEXT)' > $@.tmp
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else mv -f $@.tmp $@; fi

$(OPT068_O3FMA_STAMP): FORCE | $(OPT068_O3FMA_DIR)
	@printf '%s\n' '$(OPT068_O3FMA_FLAG_TEXT)' > $@.tmp
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else mv -f $@.tmp $@; fi

$(OPT068_O2_DIR)/q4_prompt_mmq.cuda.o: $(OPT068_FAMILY_DEPS) $(OPT068_O2_STAMP) | $(OPT068_O2_DIR)
	$(NVCC) $(OPT068_O2_NVCCFLAGS) -Icuda $(NVCC_CUDA_DEPS) -c $< -o $@

$(OPT068_O3_DIR)/q4_prompt_mmq.cuda.o: $(OPT068_FAMILY_DEPS) $(OPT068_O3_STAMP) | $(OPT068_O3_DIR)
	$(NVCC) $(OPT068_O3_NVCCFLAGS) -Icuda $(NVCC_CUDA_DEPS) -c $< -o $@

$(OPT068_O3FMA_DIR)/q4_prompt_mmq.cuda.o: $(OPT068_FAMILY_DEPS) $(OPT068_O3FMA_STAMP) | $(OPT068_O3FMA_DIR)
	$(NVCC) $(OPT068_O3FMA_NVCCFLAGS) -Icuda $(NVCC_CUDA_DEPS) -c $< -o $@

$(BUILD_DIR)/opt068_codegen_test.cuda.o: cuda/opt068_codegen_test.cu cuda/full_scheduler.h cuda/quant_mmv.h cuda/test_tier.h $(CUDA_TRACE_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_CUDA_DEPS) -c $< -o $@

$(OPT068_O2_DIR)/qw38-cuda-opt068-codegen-test: $(BUILD_DIR)/opt068_codegen_test.cuda.o $(OPT068_O2_DIR)/q4_prompt_mmq.cuda.o $(OPT068_VARIANT_LINK) $(OPT068_O2_STAMP) $(CUDA_TRACE_STAMP) | $(OPT068_O2_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_BIN_DEPS) $(BUILD_DIR)/opt068_codegen_test.cuda.o $(OPT068_O2_DIR)/q4_prompt_mmq.cuda.o $(OPT068_VARIANT_LINK) -o $@

$(OPT068_O3_DIR)/qw38-cuda-opt068-codegen-test: $(BUILD_DIR)/opt068_codegen_test.cuda.o $(OPT068_O3_DIR)/q4_prompt_mmq.cuda.o $(OPT068_VARIANT_LINK) $(OPT068_O3_STAMP) $(CUDA_TRACE_STAMP) | $(OPT068_O3_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_BIN_DEPS) $(BUILD_DIR)/opt068_codegen_test.cuda.o $(OPT068_O3_DIR)/q4_prompt_mmq.cuda.o $(OPT068_VARIANT_LINK) -o $@

$(OPT068_O3FMA_DIR)/qw38-cuda-opt068-codegen-test: $(BUILD_DIR)/opt068_codegen_test.cuda.o $(OPT068_O3FMA_DIR)/q4_prompt_mmq.cuda.o $(OPT068_VARIANT_LINK) $(OPT068_O3FMA_STAMP) $(CUDA_TRACE_STAMP) | $(OPT068_O3FMA_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_BIN_DEPS) $(BUILD_DIR)/opt068_codegen_test.cuda.o $(OPT068_O3FMA_DIR)/q4_prompt_mmq.cuda.o $(OPT068_VARIANT_LINK) -o $@

$(BUILD_DIR)/qw38-cuda-opt068-codegen-test: cuda/opt068_codegen_driver.cpp $(OPT068_O2_DIR)/qw38-cuda-opt068-codegen-test $(OPT068_O3_DIR)/qw38-cuda-opt068-codegen-test $(OPT068_O3FMA_DIR)/qw38-cuda-opt068-codegen-test | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

CUDA_STAMP_FILES := $(CUDA_STRICT_STAMP) $(CUDA_TRACE_STAMP) $(CUDA_EXPERIMENTAL_STAMP)

$(BUILD_DIR)/qw38-cuda-optimization-engine-probe: cuda/optimization_engine_probe.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(CUDA_TRACE_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_BIN_DEPS) cuda/optimization_engine_probe.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) -o $@

$(BUILD_DIR)/qw38-cuda-decode-oracle-test: cuda/decode_oracle_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(CUDA_TRACE_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_BIN_DEPS) cuda/decode_oracle_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) -o $@

$(BUILD_DIR)/qw38-cuda-prefill-4k-oracle-test: cuda/prefill_4k_oracle_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(CUDA_TRACE_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_BIN_DEPS) cuda/prefill_4k_oracle_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) -o $@

$(BUILD_DIR)/qw38-cuda-prefill-2k-parity-test: cuda/prefill_2k_parity_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(CUDA_TRACE_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_BIN_DEPS) cuda/prefill_2k_parity_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) -o $@

$(BUILD_DIR)/qw38-cuda-opt058-quality-baseline-test: cuda/opt058_quality_baseline_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(CUDA_TRACE_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_BIN_DEPS) cuda/opt058_quality_baseline_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) -o $@

$(BUILD_DIR)/qw38-cuda-opt059-numerics-test: cuda/opt059_numerics_test.cu $(QUANT_MMV_CUDA_OBJECTS) $(BUILD_DIR)/quant.o $(BUILD_DIR)/status.o $(CUDA_STRICT_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Icuda $(NVCC_BIN_DEPS) cuda/opt059_numerics_test.cu $(QUANT_MMV_CUDA_OBJECTS) $(BUILD_DIR)/quant.o $(BUILD_DIR)/status.o -o $@

$(BUILD_DIR)/qw38-cuda-opt082-kernel-parity-test: cuda/opt082_kernel_parity_test.cu cuda/kernel_parity.cuh $(QUANT_MMV_CUDA_OBJECTS) $(BUILD_DIR)/quant.o $(BUILD_DIR)/status.o $(CUDA_STRICT_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Icuda $(NVCC_BIN_DEPS) cuda/opt082_kernel_parity_test.cu $(QUANT_MMV_CUDA_OBJECTS) $(BUILD_DIR)/quant.o $(BUILD_DIR)/status.o -o $@

$(BUILD_DIR)/qw38-cuda-opt092-q8-grouped-test: cuda/opt092_q8_grouped_test.cu $(QUANT_MMV_CUDA_OBJECTS) $(BUILD_DIR)/quant.o $(BUILD_DIR)/status.o $(CUDA_STRICT_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Icuda $(NVCC_BIN_DEPS) cuda/opt092_q8_grouped_test.cu $(QUANT_MMV_CUDA_OBJECTS) $(BUILD_DIR)/quant.o $(BUILD_DIR)/status.o -o $@

$(BUILD_DIR)/qw38-cuda-opt093-q4-factored-test: cuda/opt093_q4_factored_test.cu $(QUANT_MMV_CUDA_OBJECTS) $(BUILD_DIR)/quant.o $(BUILD_DIR)/status.o $(CUDA_STRICT_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Icuda $(NVCC_BIN_DEPS) cuda/opt093_q4_factored_test.cu $(QUANT_MMV_CUDA_OBJECTS) $(BUILD_DIR)/quant.o $(BUILD_DIR)/status.o -o $@

$(BUILD_DIR)/qw38-cuda-opt100-q8-aligned-test: cuda/opt100_q8_aligned_test.cu $(QUANT_MMV_CUDA_OBJECTS) $(BUILD_DIR)/quant.o $(BUILD_DIR)/status.o $(CUDA_STRICT_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Icuda $(NVCC_BIN_DEPS) cuda/opt100_q8_aligned_test.cu $(QUANT_MMV_CUDA_OBJECTS) $(BUILD_DIR)/quant.o $(BUILD_DIR)/status.o -o $@

$(BUILD_DIR)/qw38-cuda-opt043-activation-capture-test: cuda/opt043_activation_capture_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(CUDA_TRACE_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_BIN_DEPS) cuda/opt043_activation_capture_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) -o $@

$(BUILD_DIR)/qw38-cuda-opt060-engine-attribution-test: cuda/opt060_engine_attribution_test.cu cuda/engine_attribution.h $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(CUDA_TRACE_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_BIN_DEPS) cuda/opt060_engine_attribution_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) -o $@

$(BUILD_DIR)/qw38-cuda-component-replay: cuda/optimization_component_replay.cu cuda/optimization_component_replay.h $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(CUDA_TRACE_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_BIN_DEPS) cuda/optimization_component_replay.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) -o $@

$(BUILD_DIR)/qw38-cuda-opt062-q4-admission-test: cuda/opt062_q4_admission_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(CUDA_TRACE_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_BIN_DEPS) cuda/opt062_q4_admission_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) -o $@

$(BUILD_DIR)/qw38-cuda-opt063-integer-ffn-test: cuda/opt063_integer_ffn_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(CUDA_TRACE_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_BIN_DEPS) cuda/opt063_integer_ffn_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) -o $@

$(BUILD_DIR)/qw38-cuda-opt076-q4-reduction-test: cuda/opt076_q4_reduction_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(CUDA_TRACE_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_BIN_DEPS) cuda/opt076_q4_reduction_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) -o $@

$(BUILD_DIR)/qw38-cuda-opt077-gdn-decode-test: cuda/opt077_gdn_decode_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(CUDA_TRACE_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_BIN_DEPS) cuda/opt077_gdn_decode_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) -o $@

$(BUILD_DIR)/qw38-cuda-opt101-gdn-transposed-test: cuda/opt101_gdn_transposed_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(CUDA_TRACE_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_BIN_DEPS) cuda/opt101_gdn_transposed_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) -o $@

$(BUILD_DIR)/qw38-cuda-opt102-q4-repack-test: cuda/opt102_q4_repack_test.cu $(QUANT_MMV_CUDA_OBJECTS) $(BUILD_DIR)/quant.o $(BUILD_DIR)/status.o $(CUDA_STRICT_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Icuda $(NVCC_BIN_DEPS) cuda/opt102_q4_repack_test.cu $(QUANT_MMV_CUDA_OBJECTS) $(BUILD_DIR)/quant.o $(BUILD_DIR)/status.o -o $@

$(BUILD_DIR)/qw38-cuda-opt078-decode-attention-test: cuda/opt078_decode_attention_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(CUDA_TRACE_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_BIN_DEPS) cuda/opt078_decode_attention_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) -o $@

$(BUILD_DIR)/qw38-cuda-opt095-attention-gqa-test: cuda/opt095_attention_gqa_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(CUDA_TRACE_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_BIN_DEPS) cuda/opt095_attention_gqa_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) -o $@

$(BUILD_DIR)/qw38-cuda-opt103-vector-attention-test: cuda/opt103_vector_attention_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(CUDA_TRACE_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_BIN_DEPS) cuda/opt103_vector_attention_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) -o $@

$(BUILD_DIR)/qw38-cuda-opt105-mmq-double-x-test: cuda/opt105_mmq_double_x_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(CUDA_TRACE_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_BIN_DEPS) cuda/opt105_mmq_double_x_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) -o $@

$(BUILD_DIR)/qw38-cuda-opt096-decode-graphs-test: cuda/opt096_decode_graphs_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(CUDA_TRACE_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_BIN_DEPS) cuda/opt096_decode_graphs_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) -o $@

$(BUILD_DIR)/qw38-cuda-opt097-mmq-wait-test: cuda/opt097_mmq_wait_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(CUDA_TRACE_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_BIN_DEPS) cuda/opt097_mmq_wait_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) -o $@

$(BUILD_DIR)/qw38-cuda-opt079-attention-kv-operands-test: cuda/opt079_attention_kv_operands_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(CUDA_TRACE_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_BIN_DEPS) cuda/opt079_attention_kv_operands_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) -o $@

$(BUILD_DIR)/qw38-cuda-opt064-q8-rows-test: cuda/opt064_q8_rows_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(CUDA_TRACE_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_BIN_DEPS) cuda/opt064_q8_rows_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) -o $@

$(BUILD_DIR)/qw38-cuda-opt065-mmq-tiles-test: cuda/opt065_mmq_tiles_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(CUDA_TRACE_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_BIN_DEPS) cuda/opt065_mmq_tiles_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) -o $@

$(BUILD_DIR)/qw38-cuda-opt066-mmq-x-pipeline-test: cuda/opt066_mmq_x_pipeline_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(CUDA_TRACE_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_BIN_DEPS) cuda/opt066_mmq_x_pipeline_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) -o $@

$(BUILD_DIR)/qw38-cuda-opt067-prompt-pair-test: cuda/opt067_prompt_pair_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(CUDA_TRACE_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_BIN_DEPS) cuda/opt067_prompt_pair_test.cu $(SCHEDULER_DIAGNOSTIC_CUDA_OBJECTS) $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) -o $@

$(BUILD_DIR)/qw38-cuda-probe: cuda/device_probe.cu $(CUDA_STRICT_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(NVCC_BIN_DEPS) $< -o $@

$(BUILD_DIR)/quant_mmv.cuda.o: cuda/quant_mmv.cu cuda/quant_mmv.h cuda/production_numerics.h cuda/pdl_launch.cuh cuda/q4k_decode_path.cuh cuda/q6k_decode_path.cuh cuda/ffn_decode_path.cuh $(CUDA_STRICT_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Icuda $(NVCC_CUDA_DEPS) -c $< -o $@

$(BUILD_DIR)/q4_prompt_mmq.cuda.o: cuda/q4_prompt_mmq.cu cuda/quant_mmv.h cuda/production_numerics.h cuda/quant_mmq_mma.cuh cuda/mma.cuh cuda/pdl_launch.cuh $(CUDA_STRICT_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Icuda $(NVCC_CUDA_DEPS) -c $< -o $@

$(BUILD_DIR)/q4k_decode_dots.cuda.o: cuda/q4k_decode_dots.cu cuda/q4k_decode_dots.cuh cuda/q4k_decode_path.cuh cuda/quant_mmv.h $(CUDA_STRICT_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Icuda $(NVCC_CUDA_DEPS) -c $< -o $@

$(BUILD_DIR)/q8_decode_dots.cuda.o: cuda/q8_decode_dots.cu cuda/q8_decode_dots.cuh cuda/q8_decode_path.cuh cuda/q4k_decode_dots.cuh cuda/quant_mmv.h $(CUDA_STRICT_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Icuda $(NVCC_CUDA_DEPS) -c $< -o $@

$(BUILD_DIR)/q6k_decode_dots.cuda.o: cuda/q6k_decode_dots.cu cuda/q6k_decode_dots.cuh cuda/q6k_decode_path.cuh cuda/q4k_decode_dots.cuh cuda/quant_mmv.h $(CUDA_STRICT_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Icuda $(NVCC_CUDA_DEPS) -c $< -o $@

$(BUILD_DIR)/qw38-cuda-quant-test: cuda/quant_mmv_test.cu $(QUANT_MMV_CUDA_OBJECTS) $(BUILD_DIR)/quant.o $(BUILD_DIR)/status.o $(CUDA_STRICT_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Icuda $(NVCC_BIN_DEPS) $(filter-out $(CUDA_STRICT_STAMP),$^) -o $@

$(BUILD_DIR)/qw38-cuda-dispatch-tuning-test: cuda/dispatch_tuning_test.cu $(QUANT_MMV_CUDA_OBJECTS) $(CUDA_STRICT_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Icuda $(NVCC_BIN_DEPS) $(filter-out $(CUDA_STRICT_STAMP),$^) -o $@

$(BUILD_DIR)/gdn_step.cuda.o: cuda/gdn_step.cu cuda/gdn_step.h cuda/gdn_fused_quality.cuh cuda/gdn_decode_path.cuh cuda/gdn_decode_recurrence.cuh cuda/pdl_launch.cuh $(CUDA_STRICT_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Icuda $(NVCC_CUDA_DEPS) -c $< -o $@

$(BUILD_DIR)/qw38-cuda-gdn-test: cuda/gdn_step_test.cu $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/gdn.o $(BUILD_DIR)/status.o $(CUDA_STRICT_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Icuda $(NVCC_BIN_DEPS) $(filter-out $(CUDA_STRICT_STAMP),$^) -o $@

$(BUILD_DIR)/qw38-cuda-gdn-chunk-test: cuda/gdn_chunk_test.cu $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/gdn.o $(BUILD_DIR)/status.o $(CUDA_STRICT_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Icuda $(NVCC_BIN_DEPS) $(filter-out $(CUDA_STRICT_STAMP),$^) -o $@

$(BUILD_DIR)/attention_decode.cuda.o: cuda/attention_decode.cu cuda/attention_decode.h cuda/fattn_mma_f16.cuh cuda/fattn_mma_f16_pipeline.cuh cuda/mma.cuh cuda/pdl_launch.cuh cuda/rms_norm.cuh $(CUDA_STRICT_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Icuda $(NVCC_CUDA_DEPS) -c $< -o $@

$(BUILD_DIR)/qw38-cuda-attention-test: cuda/attention_decode_test.cu $(BUILD_DIR)/attention_decode.cuda.o $(BUILD_DIR)/attention.o $(BUILD_DIR)/status.o $(CUDA_STRICT_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Icuda $(NVCC_BIN_DEPS) $(filter-out $(CUDA_STRICT_STAMP),$^) -o $@

$(BUILD_DIR)/qw38-cuda-attention-chunk-test: cuda/attention_chunk_test.cu $(BUILD_DIR)/attention_decode.cuda.o $(CUDA_STRICT_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Icuda $(NVCC_BIN_DEPS) $(filter-out $(CUDA_STRICT_STAMP),$^) -o $@

$(BUILD_DIR)/scheduler_primitives.cuda.o: cuda/scheduler_primitives.cu cuda/scheduler_primitives.h cuda/pdl_launch.cuh cuda/rms_norm.cuh $(CUDA_STRICT_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Icuda $(NVCC_CUDA_DEPS) -c $< -o $@

$(BUILD_DIR)/full_scheduler.cuda.o: cuda/full_scheduler.cu cuda/full_scheduler.h cuda/gdn_step.h cuda/gdn_decode_path.cuh cuda/pdl_launch.cuh cuda/rms_norm.cuh cuda/q8_decode_path.cuh cuda/q4k_decode_path.cuh cuda/ffn_decode_path.cuh cuda/attention_decode.h $(CUDA_STRICT_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Icuda $(NVCC_CUDA_DEPS) -c $< -o $@

$(BUILD_DIR)/full_scheduler.trace.cuda.o: cuda/full_scheduler.cu cuda/full_scheduler.h cuda/gdn_step.h cuda/gdn_decode_path.cuh cuda/pdl_launch.cuh cuda/rms_norm.cuh cuda/q8_decode_path.cuh cuda/q4k_decode_path.cuh cuda/ffn_decode_path.cuh cuda/attention_decode.h $(CUDA_TRACE_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_CUDA_DEPS) -c $< -o $@

$(BUILD_DIR)/checkpoint.cuda.o: cuda/checkpoint.cu cuda/full_scheduler.h $(CUDA_STRICT_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Icuda $(NVCC_CUDA_DEPS) -c $< -o $@

$(BUILD_DIR)/checkpoint.trace.cuda.o: cuda/checkpoint.cu cuda/full_scheduler.h $(CUDA_TRACE_STAMP) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $(NVCC_CUDA_DEPS) -c $< -o $@

$(BUILD_DIR)/qw38-cuda-scheduler-primitives-test: cuda/scheduler_primitives_test.cu $(BUILD_DIR)/scheduler_primitives.cuda.o $(QUANT_MMV_CUDA_OBJECTS) $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/quant.o $(BUILD_DIR)/status.o | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Icuda $^ -o $@

$(BUILD_DIR)/qw38-cuda-full-scheduler-test: cuda/full_scheduler_test.cu $(BUILD_DIR)/full_scheduler.trace.cuda.o $(BUILD_DIR)/scheduler_primitives.cuda.o $(QUANT_MMV_CUDA_OBJECTS) $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/attention_decode.cuda.o $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $^ -o $@

$(BUILD_DIR)/qw38-cuda-prefix-sync-test: cuda/prefix_sync_test.cu $(BUILD_DIR)/full_scheduler.trace.cuda.o $(BUILD_DIR)/scheduler_primitives.cuda.o $(QUANT_MMV_CUDA_OBJECTS) $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/attention_decode.cuda.o $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $^ -o $@

$(BUILD_DIR)/qw38-cuda-prompt-scheduler-test: cuda/prompt_scheduler_test.cu $(BUILD_DIR)/full_scheduler.trace.cuda.o $(BUILD_DIR)/scheduler_primitives.cuda.o $(QUANT_MMV_CUDA_OBJECTS) $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/attention_decode.cuda.o $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $^ -o $@

$(BUILD_DIR)/qw38-cuda-atomic-eval-test: cuda/atomic_eval_test.cu $(BUILD_DIR)/full_scheduler.trace.cuda.o $(BUILD_DIR)/scheduler_primitives.cuda.o $(QUANT_MMV_CUDA_OBJECTS) $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/attention_decode.cuda.o $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $^ -o $@

$(BUILD_DIR)/qw38-cuda-checkpoint-test: cuda/checkpoint_test.cu $(BUILD_DIR)/checkpoint.trace.cuda.o $(BUILD_DIR)/full_scheduler.trace.cuda.o $(BUILD_DIR)/scheduler_primitives.cuda.o $(QUANT_MMV_CUDA_OBJECTS) $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/attention_decode.cuda.o $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $^ -o $@

$(BUILD_DIR)/qw38-cuda-memory-fit-test: cuda/memory_fit_test.cu $(BUILD_DIR)/full_scheduler.trace.cuda.o $(BUILD_DIR)/scheduler_primitives.cuda.o $(QUANT_MMV_CUDA_OBJECTS) $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/attention_decode.cuda.o $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $^ -o $@

$(BUILD_DIR)/qw38-cuda-timing-test: cuda/timing_test.cu $(BUILD_DIR)/checkpoint.trace.cuda.o $(BUILD_DIR)/full_scheduler.trace.cuda.o $(BUILD_DIR)/scheduler_primitives.cuda.o $(QUANT_MMV_CUDA_OBJECTS) $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/attention_decode.cuda.o $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $^ -o $@

$(BUILD_DIR)/qw38-cuda-fusion-test: cuda/fusion_test.cu $(BUILD_DIR)/full_scheduler.trace.cuda.o $(BUILD_DIR)/scheduler_primitives.cuda.o $(QUANT_MMV_CUDA_OBJECTS) $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/attention_decode.cuda.o $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $^ -o $@

$(BUILD_DIR)/qw38-cuda-graph-test: cuda/graph_test.cu $(BUILD_DIR)/full_scheduler.trace.cuda.o $(BUILD_DIR)/scheduler_primitives.cuda.o $(QUANT_MMV_CUDA_OBJECTS) $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/attention_decode.cuda.o $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $^ -o $@

clean:
	rm -rf $(BUILD_DIR)

release-provenance:
	uv run python tools/release_provenance.py --require-clean

-include $(LIB_OBJECTS:.o=.d) $(DIAGNOSTIC_OBJECTS:.o=.d) $(THIRD_PARTY_OBJECTS:.o=.d) $(BUILD_DIR)/cli.d $(BUILD_DIR)/server.d $(BUILD_DIR)/bench.d $(BUILD_DIR)/eval.d $(CUDA_RELEASE_OBJECTS:.o=.d) $(CUDA_TRACE_OBJECTS:.o=.d) $(BUILD_DIR)/qw38-cuda-probe.d $(BUILD_DIR)/qw38-cuda-optimization-engine-probe.d $(BUILD_DIR)/qw38-cuda-decode-oracle-test.d $(BUILD_DIR)/qw38-cuda-prefill-4k-oracle-test.d $(BUILD_DIR)/qw38-cuda-prefill-2k-parity-test.d $(BUILD_DIR)/qw38-cuda-opt058-quality-baseline-test.d $(BUILD_DIR)/qw38-cuda-opt059-numerics-test.d $(BUILD_DIR)/qw38-cuda-opt043-activation-capture-test.d $(BUILD_DIR)/qw38-cuda-opt060-engine-attribution-test.d $(BUILD_DIR)/qw38-cuda-component-replay.d $(BUILD_DIR)/qw38-cuda-opt062-q4-admission-test.d $(BUILD_DIR)/qw38-cuda-opt063-integer-ffn-test.d $(BUILD_DIR)/qw38-cuda-opt076-q4-reduction-test.d $(BUILD_DIR)/qw38-cuda-opt077-gdn-decode-test.d $(BUILD_DIR)/qw38-cuda-opt101-gdn-transposed-test.d $(BUILD_DIR)/qw38-cuda-opt102-q4-repack-test.d $(BUILD_DIR)/qw38-cuda-opt103-vector-attention-test.d $(BUILD_DIR)/qw38-cuda-opt105-mmq-double-x-test.d $(BUILD_DIR)/qw38-cuda-opt078-decode-attention-test.d $(BUILD_DIR)/qw38-cuda-opt079-attention-kv-operands-test.d $(BUILD_DIR)/qw38-cuda-opt064-q8-rows-test.d $(BUILD_DIR)/qw38-cuda-opt065-mmq-tiles-test.d $(BUILD_DIR)/qw38-cuda-opt066-mmq-x-pipeline-test.d $(BUILD_DIR)/qw38-cuda-opt067-prompt-pair-test.d $(BUILD_DIR)/qw38-cuda-opt082-kernel-parity-test.d $(BUILD_DIR)/opt068_codegen_test.cuda.d $(wildcard $(CUDA_EXPERIMENTAL_DIR)/*.d) $(wildcard $(OPT068_O2_DIR)/*.d) $(wildcard $(OPT068_O3_DIR)/*.d) $(wildcard $(OPT068_O3FMA_DIR)/*.d)
