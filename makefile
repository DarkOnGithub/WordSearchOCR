
PROJECT_NAME = wordsearch_ocr
BUILD_DIR = build
SRC_DIR = src
TARGET = $(BUILD_DIR)/$(PROJECT_NAME)

CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -mavx2 -mfma -fopenmp -Iinclude -O3 -march=native -flto -ffast-math -funroll-loops -DNDEBUG $(GTK_CFLAGS)
LDFLAGS = -lm -fopenmp

TRAIN_CFLAGS = -O3 -march=native -flto -ffast-math -funroll-loops -mavx2 -mfma -fopenmp -DNDEBUG -Iinclude $(GTK_CFLAGS) -fopt-info-vec -fopt-info-inline
TRAIN_LDFLAGS = -O3 -flto -lm -fopenmp $(GTK_LIBS)

MAIN_SOURCES = $(shell find $(SRC_DIR) -name "*.c" -type f -not -name "*XNOR.c" -not -path "*/solver/main.c" -not -path "*/nn/train.c" -not -path "*/nn/inference_old.c")
MAIN_OBJECTS = $(MAIN_SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
DEPS = $(MAIN_OBJECTS:.o=.d)

NN_TARGET = $(BUILD_DIR)/nn/XNOR
SOLVER_TARGET = $(BUILD_DIR)/solver/solver
TRAIN_TARGET = $(BUILD_DIR)/nn/train

GTK_CFLAGS = $(shell pkg-config --cflags gtk+-3.0)
GTK_LIBS = $(shell pkg-config --libs gtk+-3.0)



all: dirs $(TARGET)

nn: dirs $(NN_TARGET)

solver: dirs $(SOLVER_TARGET)


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


$(TRAIN_TARGET): $(BUILD_DIR)/nn/train.o $(BUILD_DIR)/nn/CNN.o $(BUILD_DIR)/nn/core/tensor.o $(BUILD_DIR)/nn/core/utils.o $(BUILD_DIR)/nn/core/init.o $(BUILD_DIR)/nn/core/layer_grad.o $(BUILD_DIR)/nn/dataset/dataloader.o $(BUILD_DIR)/nn/layers/conv2D.o $(BUILD_DIR)/nn/layers/linear.o $(BUILD_DIR)/nn/layers/maxpool2D.o $(BUILD_DIR)/nn/layers/adaptive_avg_pool2D.o $(BUILD_DIR)/nn/layers/dropout.o $(BUILD_DIR)/nn/layers/dropout2D.o $(BUILD_DIR)/nn/layers/batch_norm2d.o $(BUILD_DIR)/nn/layers/cross_entropy_loss.o $(BUILD_DIR)/nn/nn/adamW.o $(BUILD_DIR)/nn/nn/step_lr_scheduler.o $(BUILD_DIR)/nn/nn/silu.o $(BUILD_DIR)/nn/nn/relu.o $(BUILD_DIR)/image/image.o
	@echo "Linking optimized train $@..."
	$(CC) $^ -o $@ $(TRAIN_LDFLAGS)
	@echo "Optimized train build successful! Binary: $@"

train: dirs $(TRAIN_TARGET)
	@echo "Built with aggressive optimizations for maximum training performance!"
	@echo "Flags used: $(TRAIN_CFLAGS)"
	@echo "Run with: ./$(TRAIN_TARGET)"
	@echo "For maximum CPU utilization, run with:"
	@echo "OMP_NUM_THREADS=$(shell nproc) ./$(TRAIN_TARGET)"



$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@echo "Compiling $<..."
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(GTK_CFLAGS) -MMD -MP -c $< -o $@

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


-include $(DEPS)


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
	@echo "Compiler: $(CC)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "LDFLAGS: $(LDFLAGS)"

help:
	@echo "=== $(PROJECT_NAME) Makefile Help ==="
	@echo ""
	@echo "Available targets:"
	@echo "  all             - Build the project with optimizations (default)"
	@echo "  nn              - Build the XNOR neural network example"
	@echo "  solver          - Build the word search solver"
	@echo "  train           - Build the neural network training program"
	@echo "  clean           - Remove all build artifacts"
	@echo "  release         - Build optimized release"
	@echo "  info            - Show build information"
	@echo ""
	@echo "Build directory: $(BUILD_DIR)"
	@echo "Source directory: $(SRC_DIR)"

.PHONY: all nn solver train clean release dirs info help
