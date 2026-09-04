UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S),Darwin)
CXX ?= clang++
CC ?= clang
else
CXX ?= g++
CC ?= cc
endif
NVCC ?= nvcc
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror -fno-exceptions -fno-rtti -ffp-contract=off -pthread
ifeq ($(UNAME_S),Darwin)
ifeq ($(UNAME_M),x86_64)
CXXFLAGS += -mavx2 -mfma
endif
endif
CFLAGS := -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror
NVCCFLAGS := -std=c++17 -O2 -arch=sm_120 --expt-relaxed-constexpr --fmad=false -Xcompiler=-Wall,-Wextra,-Werror,-fno-exceptions,-fno-rtti,-ffp-contract=off,-pthread
CPPFLAGS := -Iinclude -Isrc -Ithird_party/utf8proc
BUILD_DIR := build
CUDA_BUILD_DIR := $(BUILD_DIR)/cuda
DIAGNOSTIC_DIR := $(BUILD_DIR)/diagnostic
LIB_SOURCES := src/status.cpp src/sha256.cpp src/model.cpp src/tokenizer.cpp src/template.cpp src/quant.cpp src/tensor.cpp src/conversion.cpp src/projection.cpp src/weights.cpp src/mixer.cpp src/scheduler.cpp src/scalar_runtime.cpp src/gdn.cpp src/attention.cpp src/host_checkpoint.cpp src/engine.cpp
LIB_OBJECTS := $(LIB_SOURCES:src/%.cpp=$(BUILD_DIR)/%.o)
THIRD_PARTY_OBJECTS := $(BUILD_DIR)/utf8proc.o
BINARIES := $(BUILD_DIR)/qw38 $(BUILD_DIR)/qw38-server $(BUILD_DIR)/qw38-bench $(BUILD_DIR)/qw38-eval
HOST_DIAGNOSTICS := $(BUILD_DIR)/qw38-server-core-test $(BUILD_DIR)/qw38-server-api-test
CUDA_IMAGE := qw38-cuda:13.0.2

.PHONY: all clean test diagnostic cuda-image cuda-build cuda-native cuda-products

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

SERVER_OBJECTS := $(BUILD_DIR)/server_core.o $(BUILD_DIR)/server_json.o $(BUILD_DIR)/server_api.o $(BUILD_DIR)/server_generation.o

$(BUILD_DIR)/qw38-server: $(LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(SERVER_OBJECTS) $(BUILD_DIR)/server.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/qw38-server-core-test: $(BUILD_DIR)/status.o $(BUILD_DIR)/server_core.o $(BUILD_DIR)/server_core_test.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/qw38-server-api-test: $(BUILD_DIR)/status.o $(THIRD_PARTY_OBJECTS) $(BUILD_DIR)/server_json.o $(BUILD_DIR)/server_api.o $(BUILD_DIR)/server_api_test.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/qw38-bench: $(LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(BUILD_DIR)/bench.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/qw38-eval: $(LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(BUILD_DIR)/eval.o
	$(CXX) $(CXXFLAGS) $^ -o $@

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

$(CUDA_BUILD_DIR)/engine.o: src/engine.cpp include/qw38/engine.h cuda/full_scheduler.h | $(CUDA_BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_CUDA_RUNTIME -Icuda -c $< -o $@

CUDA_ENGINE_OBJECTS := $(filter-out $(BUILD_DIR)/engine.o,$(LIB_OBJECTS)) $(CUDA_BUILD_DIR)/engine.o $(THIRD_PARTY_OBJECTS)

cuda-products: $(CUDA_BUILD_DIR)/qw38 $(CUDA_BUILD_DIR)/qw38-server

$(CUDA_BUILD_DIR)/qw38: $(CUDA_ENGINE_OBJECTS) $(BUILD_DIR)/cli.o $(BUILD_DIR)/checkpoint.cuda.o $(BUILD_DIR)/full_scheduler.cuda.o $(BUILD_DIR)/scheduler_primitives.cuda.o $(BUILD_DIR)/quant_mmv.cuda.o $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/attention_decode.cuda.o | $(CUDA_BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(CUDA_BUILD_DIR)/qw38-server: $(CUDA_ENGINE_OBJECTS) $(SERVER_OBJECTS) $(BUILD_DIR)/server.o $(BUILD_DIR)/checkpoint.cuda.o $(BUILD_DIR)/full_scheduler.cuda.o $(BUILD_DIR)/scheduler_primitives.cuda.o $(BUILD_DIR)/quant_mmv.cuda.o $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/attention_decode.cuda.o | $(CUDA_BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

cuda-native: $(BUILD_DIR)/qw38-cuda-probe $(BUILD_DIR)/qw38-cuda-quant-test $(BUILD_DIR)/qw38-cuda-dispatch-tuning-test $(BUILD_DIR)/qw38-cuda-gdn-test $(BUILD_DIR)/qw38-cuda-gdn-chunk-test $(BUILD_DIR)/qw38-cuda-attention-test $(BUILD_DIR)/qw38-cuda-attention-chunk-test $(BUILD_DIR)/qw38-cuda-scheduler-primitives-test $(BUILD_DIR)/qw38-cuda-full-scheduler-test $(BUILD_DIR)/qw38-cuda-prefix-sync-test $(BUILD_DIR)/qw38-cuda-atomic-eval-test $(BUILD_DIR)/qw38-cuda-checkpoint-test $(BUILD_DIR)/qw38-cuda-memory-fit-test $(BUILD_DIR)/qw38-cuda-timing-test $(BUILD_DIR)/qw38-cuda-fusion-test $(BUILD_DIR)/qw38-cuda-graph-test

$(BUILD_DIR)/qw38-cuda-probe: cuda/device_probe.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD_DIR)/quant_mmv.cuda.o: cuda/quant_mmv.cu cuda/quant_mmv.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Icuda -c $< -o $@

$(BUILD_DIR)/qw38-cuda-quant-test: cuda/quant_mmv_test.cu $(BUILD_DIR)/quant_mmv.cuda.o $(BUILD_DIR)/quant.o $(BUILD_DIR)/status.o | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Icuda $^ -o $@

$(BUILD_DIR)/qw38-cuda-dispatch-tuning-test: cuda/dispatch_tuning_test.cu $(BUILD_DIR)/quant_mmv.cuda.o | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Icuda $^ -o $@

$(BUILD_DIR)/gdn_step.cuda.o: cuda/gdn_step.cu cuda/gdn_step.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Icuda -c $< -o $@

$(BUILD_DIR)/qw38-cuda-gdn-test: cuda/gdn_step_test.cu $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/gdn.o $(BUILD_DIR)/status.o | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Icuda $^ -o $@

$(BUILD_DIR)/qw38-cuda-gdn-chunk-test: cuda/gdn_chunk_test.cu $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/gdn.o $(BUILD_DIR)/status.o | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Icuda $^ -o $@

$(BUILD_DIR)/attention_decode.cuda.o: cuda/attention_decode.cu cuda/attention_decode.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Icuda -c $< -o $@

$(BUILD_DIR)/qw38-cuda-attention-test: cuda/attention_decode_test.cu $(BUILD_DIR)/attention_decode.cuda.o $(BUILD_DIR)/attention.o $(BUILD_DIR)/status.o | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Icuda $^ -o $@

$(BUILD_DIR)/qw38-cuda-attention-chunk-test: cuda/attention_chunk_test.cu $(BUILD_DIR)/attention_decode.cuda.o | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Icuda $^ -o $@

$(BUILD_DIR)/scheduler_primitives.cuda.o: cuda/scheduler_primitives.cu cuda/scheduler_primitives.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Icuda -c $< -o $@

$(BUILD_DIR)/full_scheduler.cuda.o: cuda/full_scheduler.cu cuda/full_scheduler.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Icuda -c $< -o $@

$(BUILD_DIR)/checkpoint.cuda.o: cuda/checkpoint.cu cuda/full_scheduler.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Icuda -c $< -o $@

$(BUILD_DIR)/qw38-cuda-scheduler-primitives-test: cuda/scheduler_primitives_test.cu $(BUILD_DIR)/scheduler_primitives.cuda.o $(BUILD_DIR)/quant_mmv.cuda.o $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/quant.o $(BUILD_DIR)/status.o | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Icuda $^ -o $@

$(BUILD_DIR)/qw38-cuda-full-scheduler-test: cuda/full_scheduler_test.cu $(BUILD_DIR)/full_scheduler.cuda.o $(BUILD_DIR)/scheduler_primitives.cuda.o $(BUILD_DIR)/quant_mmv.cuda.o $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/attention_decode.cuda.o $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $^ -o $@

$(BUILD_DIR)/qw38-cuda-prefix-sync-test: cuda/prefix_sync_test.cu $(BUILD_DIR)/full_scheduler.cuda.o $(BUILD_DIR)/scheduler_primitives.cuda.o $(BUILD_DIR)/quant_mmv.cuda.o $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/attention_decode.cuda.o $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $^ -o $@

$(BUILD_DIR)/qw38-cuda-atomic-eval-test: cuda/atomic_eval_test.cu $(BUILD_DIR)/full_scheduler.cuda.o $(BUILD_DIR)/scheduler_primitives.cuda.o $(BUILD_DIR)/quant_mmv.cuda.o $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/attention_decode.cuda.o $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $^ -o $@

$(BUILD_DIR)/qw38-cuda-checkpoint-test: cuda/checkpoint_test.cu $(BUILD_DIR)/checkpoint.cuda.o $(BUILD_DIR)/full_scheduler.cuda.o $(BUILD_DIR)/scheduler_primitives.cuda.o $(BUILD_DIR)/quant_mmv.cuda.o $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/attention_decode.cuda.o $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $^ -o $@

$(BUILD_DIR)/qw38-cuda-memory-fit-test: cuda/memory_fit_test.cu $(BUILD_DIR)/full_scheduler.cuda.o $(BUILD_DIR)/scheduler_primitives.cuda.o $(BUILD_DIR)/quant_mmv.cuda.o $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/attention_decode.cuda.o $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $^ -o $@

$(BUILD_DIR)/qw38-cuda-timing-test: cuda/timing_test.cu $(BUILD_DIR)/checkpoint.cuda.o $(BUILD_DIR)/full_scheduler.cuda.o $(BUILD_DIR)/scheduler_primitives.cuda.o $(BUILD_DIR)/quant_mmv.cuda.o $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/attention_decode.cuda.o $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $^ -o $@

$(BUILD_DIR)/qw38-cuda-fusion-test: cuda/fusion_test.cu $(BUILD_DIR)/full_scheduler.cuda.o $(BUILD_DIR)/scheduler_primitives.cuda.o $(BUILD_DIR)/quant_mmv.cuda.o $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/attention_decode.cuda.o $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $^ -o $@

$(BUILD_DIR)/qw38-cuda-graph-test: cuda/graph_test.cu $(BUILD_DIR)/full_scheduler.cuda.o $(BUILD_DIR)/scheduler_primitives.cuda.o $(BUILD_DIR)/quant_mmv.cuda.o $(BUILD_DIR)/gdn_step.cuda.o $(BUILD_DIR)/attention_decode.cuda.o $(DIAGNOSTIC_LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -DQW38_DIAGNOSTIC_TRACE -Icuda $^ -o $@

clean:
	rm -rf $(BUILD_DIR)

-include $(LIB_OBJECTS:.o=.d) $(DIAGNOSTIC_OBJECTS:.o=.d) $(THIRD_PARTY_OBJECTS:.o=.d) $(BUILD_DIR)/cli.d $(BUILD_DIR)/server.d $(BUILD_DIR)/bench.d $(BUILD_DIR)/eval.d
