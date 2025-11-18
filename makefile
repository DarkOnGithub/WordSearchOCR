
PROJECT_NAME = wordsearch_ocr
BUILD_DIR = build
SRC_DIR = src
TARGET = $(BUILD_DIR)/$(PROJECT_NAME)

CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -g -mavx2 -mfma -fopenmp -Iinclude
LDFLAGS = -lm -fopenmp

# Training-specific optimization flags for maximum performance
TRAIN_CFLAGS = -O3 -march=native -flto -ffast-math -funroll-loops -mavx2 -mfma -fopenmp -DNDEBUG -Iinclude $(GTK_CFLAGS) -fopt-info-vec -fopt-info-inline
TRAIN_LDFLAGS = -O3 -flto -lm -fopenmp $(GTK_LIBS)

# Profiling flags for gprof
PROFILE_CFLAGS = -pg -g -O2 -mavx2 -mfma -Iinclude
PROFILE_LDFLAGS = -pg -lm

MAIN_SOURCES = $(shell find $(SRC_DIR) -name "*.c" -type f -not -name "*XNOR.c" -not -path "*/solver/main.c" -not -path "*/nn/train.c" -not -path "*/nn/inference_old.c")
MAIN_OBJECTS = $(MAIN_SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
DEPS = $(MAIN_OBJECTS:.o=.d)

NN_TARGET = $(BUILD_DIR)/nn/XNOR
SOLVER_TARGET = $(BUILD_DIR)/solver/solver
TEST_TARGET = $(BUILD_DIR)/test/test
BENCHMARK_TARGET = $(BUILD_DIR)/benchmark_linear
TRAIN_TARGET = $(BUILD_DIR)/nn/train
INFERENCE_TARGET = $(BUILD_DIR)/nn/inference

GTK3_AVAILABLE := $(shell pkg-config --exists gtk+-3.0 2>/dev/null && echo "yes" || echo "no")

ifeq ($(GTK3_AVAILABLE),yes)
    GTK_CFLAGS = $(shell pkg-config --cflags gtk+-3.0)
    GTK_LIBS = $(shell pkg-config --libs gtk+-3.0)
else
    GTK_CFLAGS =
    GTK_LIBS =
endif



all: dirs $(TARGET)

nn: dirs $(NN_TARGET)

solver: dirs $(SOLVER_TARGET)

test: dirs $(TEST_TARGET)

benchmark: dirs $(BENCHMARK_TARGET)

run-benchmark: benchmark
	@echo "Running linear layer benchmark..."
	@./$(BENCHMARK_TARGET)

run-test: test
	@echo "Running tensor tests..."
	@./$(TEST_TARGET)

check-deps:
	@if [ "$(GTK3_AVAILABLE)" != "yes" ]; then \
		echo "Error: Missing required dependencies."; \
		echo "Required: libgtk-3-dev"; \
		echo "Run 'make install-deps' to install them."; \
		exit 1; \
	fi
	@echo "Dependencies OK: GTK3=$(GTK3_AVAILABLE)"

dirs:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BUILD_DIR)/analysis
	@mkdir -p $(BUILD_DIR)/image
	@mkdir -p $(BUILD_DIR)/detection
	@mkdir -p $(BUILD_DIR)/preprocessing
	@mkdir -p $(BUILD_DIR)/wordsearch
	@mkdir -p $(BUILD_DIR)/gui
	@mkdir -p $(BUILD_DIR)/nn
	@mkdir -p $(BUILD_DIR)/solver
	@mkdir -p $(BUILD_DIR)/test

$(TARGET): $(MAIN_OBJECTS)
	@echo "Linking $@..."
	$(CC) $(MAIN_OBJECTS) -o $@ $(GTK_LIBS) $(LDFLAGS)
	@echo "Build successful! Binary: $@"

$(NN_TARGET): $(BUILD_DIR)/nn/XNOR.o
	@echo "Linking neural network $@..."
	$(CC) $(BUILD_DIR)/nn/XNOR.o -o $@ $(LDFLAGS)
	@echo "Neural network build successful! Binary: $@"

$(SOLVER_TARGET): $(BUILD_DIR)/solver/main.o $(BUILD_DIR)/solver/solver.o $(BUILD_DIR)/solver/search.o
	@echo "Linking solver $@..."
	$(CC) $(BUILD_DIR)/solver/main.o $(BUILD_DIR)/solver/solver.o $(BUILD_DIR)/solver/search.o -o $@ $(LDFLAGS)
	@echo "Solver build successful! Binary: $@"

$(TEST_TARGET): $(BUILD_DIR)/test/test.o $(BUILD_DIR)/nn/core/tensor.o $(BUILD_DIR)/nn/core/utils.o $(BUILD_DIR)/nn/core/init.o
	@echo "Linking test $@..."
	$(CC) $(BUILD_DIR)/test/test.o $(BUILD_DIR)/nn/core/tensor.o $(BUILD_DIR)/nn/core/utils.o $(BUILD_DIR)/nn/core/init.o -o $@ $(LDFLAGS)
	@echo "Test build successful! Binary: $@"

$(BENCHMARK_TARGET): $(BUILD_DIR)/test/benchmark_linear.o $(BUILD_DIR)/nn/core/tensor.o $(BUILD_DIR)/nn/core/utils.o $(BUILD_DIR)/nn/core/init.o $(BUILD_DIR)/nn/core/layer_grad.o $(BUILD_DIR)/nn/layers/linear.o
	@echo "Linking benchmark $@..."
	$(CC) $(BUILD_DIR)/test/benchmark_linear.o $(BUILD_DIR)/nn/core/tensor.o $(BUILD_DIR)/nn/core/utils.o $(BUILD_DIR)/nn/core/init.o $(BUILD_DIR)/nn/core/layer_grad.o $(BUILD_DIR)/nn/layers/linear.o -o $@ $(LDFLAGS)
	@echo "Benchmark build successful! Binary: $@"

$(TRAIN_TARGET): $(BUILD_DIR)/nn/train.o $(BUILD_DIR)/nn/CNN.o $(BUILD_DIR)/nn/core/tensor.o $(BUILD_DIR)/nn/core/utils.o $(BUILD_DIR)/nn/core/init.o $(BUILD_DIR)/nn/core/layer_grad.o $(BUILD_DIR)/nn/dataset/dataloader.o $(BUILD_DIR)/nn/layers/conv2D.o $(BUILD_DIR)/nn/layers/linear.o $(BUILD_DIR)/nn/layers/maxpool2D.o $(BUILD_DIR)/nn/layers/adaptive_avg_pool2D.o $(BUILD_DIR)/nn/layers/dropout.o $(BUILD_DIR)/nn/layers/dropout2D.o $(BUILD_DIR)/nn/layers/batch_norm2d.o $(BUILD_DIR)/nn/layers/cross_entropy_loss.o $(BUILD_DIR)/nn/nn/adamW.o $(BUILD_DIR)/nn/nn/step_lr_scheduler.o $(BUILD_DIR)/nn/nn/silu.o $(BUILD_DIR)/nn/nn/relu.o $(BUILD_DIR)/image/image.o
	@echo "Linking optimized train $@..."
	$(CC) $^ -o $@ $(TRAIN_LDFLAGS)
	@echo "Optimized train build successful! Binary: $@"

$(INFERENCE_TARGET): $(BUILD_DIR)/nn/inference.o $(BUILD_DIR)/nn/CNN.o $(BUILD_DIR)/nn/core/tensor.o $(BUILD_DIR)/nn/core/utils.o $(BUILD_DIR)/nn/core/init.o $(BUILD_DIR)/nn/core/layer_grad.o $(BUILD_DIR)/nn/layers/conv2D.o $(BUILD_DIR)/nn/layers/linear.o $(BUILD_DIR)/nn/layers/maxpool2D.o $(BUILD_DIR)/nn/layers/adaptive_avg_pool2D.o $(BUILD_DIR)/nn/layers/dropout.o $(BUILD_DIR)/nn/layers/dropout2D.o $(BUILD_DIR)/nn/layers/batch_norm2d.o $(BUILD_DIR)/nn/layers/cross_entropy_loss.o $(BUILD_DIR)/nn/nn/adamW.o $(BUILD_DIR)/nn/nn/step_lr_scheduler.o $(BUILD_DIR)/nn/nn/silu.o $(BUILD_DIR)/nn/nn/relu.o $(BUILD_DIR)/image/image.o $(BUILD_DIR)/image/operations.o
	@echo "Linking inference $@..."
	$(CC) $^ -o $@ $(TRAIN_LDFLAGS)
	@echo "Inference build successful! Binary: $@"

train: dirs $(TRAIN_TARGET)

inference: dirs $(INFERENCE_TARGET)

train-optimized: dirs $(TRAIN_TARGET)
	@echo "Built with aggressive optimizations for maximum training performance!"
	@echo "Flags used: $(TRAIN_CFLAGS)"
	@echo "Run with: ./$(TRAIN_TARGET)"
	@echo "For maximum CPU utilization, run with:"
	@echo "OMP_NUM_THREADS=$(shell nproc) ./$(TRAIN_TARGET)"

# Profiling targets
PROFILE_TRAIN_TARGET = $(BUILD_DIR)/profile_train
PROFILE_NN_TARGET = $(BUILD_DIR)/profile_nn

profile-train: dirs $(PROFILE_TRAIN_TARGET)
	@echo "Built with profiling enabled for training!"
	@echo "Run with: ./$(PROFILE_TRAIN_TARGET)"
	@echo "Then run: gprof $(PROFILE_TRAIN_TARGET) gmon.out > profile_train.txt"

profile-nn: dirs $(PROFILE_NN_TARGET)
	@echo "Built with profiling enabled for neural network example!"
	@echo "Run with: ./$(PROFILE_NN_TARGET)"
	@echo "Then run: gprof $(PROFILE_NN_TARGET) gmon.out > profile_nn.txt"

$(PROFILE_TRAIN_TARGET): $(BUILD_DIR)/nn/profile_train.o $(BUILD_DIR)/nn/profile_CNN.o $(BUILD_DIR)/nn/core/profile_tensor.o $(BUILD_DIR)/nn/core/profile_utils.o $(BUILD_DIR)/nn/core/profile_init.o $(BUILD_DIR)/nn/core/profile_layer_grad.o $(BUILD_DIR)/nn/dataset/profile_dataloader.o $(BUILD_DIR)/nn/layers/profile_conv2D.o $(BUILD_DIR)/nn/layers/profile_linear.o $(BUILD_DIR)/nn/layers/profile_maxpool2D.o $(BUILD_DIR)/nn/layers/profile_dropout.o $(BUILD_DIR)/nn/layers/profile_dropout2D.o $(BUILD_DIR)/nn/layers/profile_cross_entropy_loss.o $(BUILD_DIR)/nn/nn/profile_adamW.o $(BUILD_DIR)/nn/nn/profile_step_lr_scheduler.o $(BUILD_DIR)/nn/nn/profile_relu.o
	@echo "Linking profiled train $@..."
	$(CC) $^ -o $@ $(PROFILE_LDFLAGS)

$(PROFILE_NN_TARGET): $(BUILD_DIR)/nn/profile_XNOR.o
	@echo "Linking profiled neural network $@..."
	$(CC) $(BUILD_DIR)/nn/profile_XNOR.o -o $@ $(PROFILE_LDFLAGS)


$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@echo "Compiling $<..."
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(GTK_CFLAGS) -MMD -MP -c $< -o $@

# Special compilation rules for training-critical objects with aggressive optimization
$(BUILD_DIR)/nn/train.o: $(SRC_DIR)/nn/train.c
	@echo "Compiling optimized $<..."
	@mkdir -p $(dir $@)
	$(CC) $(TRAIN_CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/nn/CNN.o: $(SRC_DIR)/nn/CNN.c
	@echo "Compiling optimized $<..."
	@mkdir -p $(dir $@)
	$(CC) $(TRAIN_CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/nn/layers/conv2D.o: $(SRC_DIR)/nn/layers/conv2D.c
	@echo "Compiling optimized $<..."
	@mkdir -p $(dir $@)
	$(CC) $(TRAIN_CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/nn/layers/linear.o: $(SRC_DIR)/nn/layers/linear.c
	@echo "Compiling optimized $<..."
	@mkdir -p $(dir $@)
	$(CC) $(TRAIN_CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/nn/nn/adamW.o: $(SRC_DIR)/nn/nn/adamW.c
	@echo "Compiling optimized $<..."
	@mkdir -p $(dir $@)
	$(CC) $(TRAIN_CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/nn/core/tensor.o: $(SRC_DIR)/nn/core/tensor.c
	@echo "Compiling optimized $<..."
	@mkdir -p $(dir $@)
	$(CC) $(TRAIN_CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/nn/core/layer_grad.o: $(SRC_DIR)/nn/core/layer_grad.c
	@echo "Compiling optimized $<..."
	@mkdir -p $(dir $@)
	$(CC) $(TRAIN_CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/nn/layers/cross_entropy_loss.o: $(SRC_DIR)/nn/layers/cross_entropy_loss.c
	@echo "Compiling optimized $<..."
	@mkdir -p $(dir $@)
	$(CC) $(TRAIN_CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/test/%.o: test/%.c
	@echo "Compiling test $<..."
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

# Profiling compilation rules
$(BUILD_DIR)/nn/profile_%.o: $(SRC_DIR)/nn/%.c
	@echo "Compiling profiled $<..."
	@mkdir -p $(dir $@)
	$(CC) $(PROFILE_CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/nn/core/profile_%.o: $(SRC_DIR)/nn/core/%.c
	@echo "Compiling profiled $<..."
	@mkdir -p $(dir $@)
	$(CC) $(PROFILE_CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/nn/dataset/profile_%.o: $(SRC_DIR)/nn/dataset/%.c
	@echo "Compiling profiled $<..."
	@mkdir -p $(dir $@)
	$(CC) $(PROFILE_CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/nn/layers/profile_%.o: $(SRC_DIR)/nn/layers/%.c
	@echo "Compiling profiled $<..."
	@mkdir -p $(dir $@)
	$(CC) $(PROFILE_CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/nn/nn/profile_%.o: $(SRC_DIR)/nn/nn/%.c
	@echo "Compiling profiled $<..."
	@mkdir -p $(dir $@)
	$(CC) $(PROFILE_CFLAGS) -MMD -MP -c $< -o $@

-include $(DEPS)

install-deps:
	@echo "Installing GTK3 development libraries..."
	@command -v apt-get >/dev/null 2>&1 && { \
		sudo apt-get update && \
		sudo apt-get install -y libgtk-3-dev; \
	} || { \
		echo "Error: apt-get not found."; \
		echo "Please install GTK3 manually."; \
		exit 1; \
	}

clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)

release: CFLAGS += -O3 -DNDEBUG
release: all

info:
	@echo "=== Build Information ==="
	@echo "Project: $(PROJECT_NAME)"
	@echo "Target: $(TARGET)"
	@echo "Main Sources: $(words $(MAIN_SOURCES))"
	@echo "Main Objects: $(words $(MAIN_OBJECTS))"
	@echo "GTK3: $(GTK3_AVAILABLE)"
	@echo "Compiler: $(CC)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "LDFLAGS: $(LDFLAGS)"

help:
	@echo "=== $(PROJECT_NAME) Makefile Help ==="
	@echo ""
	@echo "Available targets:"
	@echo "  all             - Build the project (default)"
	@echo "  nn              - Build the XNOR neural network example"
	@echo "  solver          - Build the word search solver"
	@echo "  test            - Build tensor tests"
	@echo "  benchmark       - Build linear layer benchmark"
	@echo "  run-benchmark   - Build and run linear layer benchmark"
	@echo "  run-test        - Build and run tensor tests"
	@echo "  train           - Build the neural network training program"
	@echo "  inference       - Build the neural network inference program"
	@echo "  train-optimized - Build training program with aggressive optimizations"
	@echo "  profile-train   - Build training program with gprof profiling enabled"
	@echo "  profile-nn      - Build neural network example with gprof profiling enabled"
	@echo "  clean           - Remove all build artifacts"
	@echo "  release         - Build optimized release"
	@echo "  install-deps    - Install GTK3 dependencies"
	@echo "  check-deps      - Check if dependencies are installed"
	@echo "  info            - Show build information"
	@echo ""
	@echo "Build directory: $(BUILD_DIR)"
	@echo "Source directory: $(SRC_DIR)"

.PHONY: all nn solver test benchmark run-benchmark run-test train inference train-optimized profile-train profile-nn clean release install-deps check-deps dirs info help
