CXX ?= g++
CC ?= cc
NVCC ?= nvcc
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror -fno-exceptions -fno-rtti -ffp-contract=off
CFLAGS := -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror
NVCCFLAGS := -std=c++17 -O2 -arch=sm_120 --expt-relaxed-constexpr --fmad=false -Xcompiler=-Wall,-Wextra,-Werror,-fno-exceptions,-fno-rtti,-ffp-contract=off
CPPFLAGS := -Iinclude -Isrc -Ithird_party/utf8proc
BUILD_DIR := build
DIAGNOSTIC_DIR := $(BUILD_DIR)/diagnostic
LIB_SOURCES := src/status.cpp src/sha256.cpp src/model.cpp src/tokenizer.cpp src/template.cpp src/quant.cpp src/tensor.cpp src/conversion.cpp src/projection.cpp src/weights.cpp src/mixer.cpp src/scheduler.cpp src/scalar_runtime.cpp src/gdn.cpp src/attention.cpp src/engine.cpp
LIB_OBJECTS := $(LIB_SOURCES:src/%.cpp=$(BUILD_DIR)/%.o)
THIRD_PARTY_OBJECTS := $(BUILD_DIR)/utf8proc.o
BINARIES := $(BUILD_DIR)/qw38 $(BUILD_DIR)/qw38-server $(BUILD_DIR)/qw38-bench $(BUILD_DIR)/qw38-eval
CUDA_IMAGE := qw38-cuda:13.0.2

.PHONY: all clean test diagnostic cuda-image cuda-build cuda-native

all: $(BINARIES)

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

$(BUILD_DIR)/qw38-server: $(LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(BUILD_DIR)/server.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/qw38-bench: $(LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(BUILD_DIR)/bench.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/qw38-eval: $(LIB_OBJECTS) $(THIRD_PARTY_OBJECTS) $(BUILD_DIR)/eval.o
	$(CXX) $(CXXFLAGS) $^ -o $@

DIAGNOSTIC_OBJECTS := $(LIB_SOURCES:src/%.cpp=$(DIAGNOSTIC_DIR)/%.o) $(DIAGNOSTIC_DIR)/diagnostic_trace.o $(DIAGNOSTIC_DIR)/eval.o

diagnostic: $(BUILD_DIR)/qw38-eval-diagnostic

$(BUILD_DIR)/qw38-eval-diagnostic: $(DIAGNOSTIC_OBJECTS) $(THIRD_PARTY_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

test: all
	uv run pytest

cuda-image:
	docker build -f docker/cuda.Dockerfile -t $(CUDA_IMAGE) .

cuda-build: cuda-image
	docker run --rm --gpus all --user "$$(id -u):$$(id -g)" -v "$$(pwd):/workspace" $(CUDA_IMAGE) make clean all cuda-native

cuda-native: $(BUILD_DIR)/qw38-cuda-probe $(BUILD_DIR)/qw38-cuda-quant-test $(BUILD_DIR)/qw38-cuda-gdn-test $(BUILD_DIR)/qw38-cuda-gdn-chunk-test $(BUILD_DIR)/qw38-cuda-attention-test $(BUILD_DIR)/qw38-cuda-attention-chunk-test

$(BUILD_DIR)/qw38-cuda-probe: cuda/device_probe.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD_DIR)/quant_mmv.cuda.o: cuda/quant_mmv.cu cuda/quant_mmv.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Icuda -c $< -o $@

$(BUILD_DIR)/qw38-cuda-quant-test: cuda/quant_mmv_test.cu $(BUILD_DIR)/quant_mmv.cuda.o $(BUILD_DIR)/quant.o $(BUILD_DIR)/status.o | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Icuda $^ -o $@

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

clean:
	rm -rf $(BUILD_DIR)

-include $(LIB_OBJECTS:.o=.d) $(DIAGNOSTIC_OBJECTS:.o=.d) $(THIRD_PARTY_OBJECTS:.o=.d) $(BUILD_DIR)/cli.d $(BUILD_DIR)/server.d $(BUILD_DIR)/bench.d $(BUILD_DIR)/eval.d
