CXX ?= g++
NVCC ?= nvcc
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror -fno-exceptions -fno-rtti
NVCCFLAGS := -std=c++17 -O2 -arch=sm_120 --expt-relaxed-constexpr
CPPFLAGS := -Iinclude -Isrc
BUILD_DIR := build
LIB_SOURCES := src/status.cpp src/sha256.cpp src/model.cpp src/engine.cpp
LIB_OBJECTS := $(LIB_SOURCES:src/%.cpp=$(BUILD_DIR)/%.o)
BINARIES := $(BUILD_DIR)/qw38 $(BUILD_DIR)/qw38-server $(BUILD_DIR)/qw38-bench $(BUILD_DIR)/qw38-eval
CUDA_IMAGE := qw38-cuda:13.0.2

.PHONY: all clean test cuda-image cuda-build cuda-native

all: $(BINARIES)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/qw38: $(LIB_OBJECTS) $(BUILD_DIR)/cli.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/qw38-server: $(LIB_OBJECTS) $(BUILD_DIR)/server.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/qw38-bench: $(LIB_OBJECTS) $(BUILD_DIR)/bench.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/qw38-eval: $(LIB_OBJECTS) $(BUILD_DIR)/eval.o
	$(CXX) $(CXXFLAGS) $^ -o $@

test: all
	uv run pytest

cuda-image:
	docker build -f docker/cuda.Dockerfile -t $(CUDA_IMAGE) .

cuda-build: cuda-image
	docker run --rm --gpus all --user "$$(id -u):$$(id -g)" -v "$$(pwd):/workspace" $(CUDA_IMAGE) make clean all cuda-native

cuda-native: $(BUILD_DIR)/qw38-cuda-probe

$(BUILD_DIR)/qw38-cuda-probe: cuda/device_probe.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -rf $(BUILD_DIR)

-include $(LIB_OBJECTS:.o=.d) $(BUILD_DIR)/cli.d $(BUILD_DIR)/server.d $(BUILD_DIR)/bench.d $(BUILD_DIR)/eval.d
