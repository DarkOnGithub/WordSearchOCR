
PROJECT_NAME = wordsearch_ocr
BUILD_DIR = build
SRC_DIR = src
TARGET = $(BUILD_DIR)/$(PROJECT_NAME)

CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -g -Iinclude
LDFLAGS = -lm

MAIN_SOURCES = $(shell find $(SRC_DIR) -name "*.c" -type f -not -name "*XNOR.c" -not -path "*/solver/main.c" -not -path "*/tests/*")
MAIN_OBJECTS = $(MAIN_SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
DEPS = $(MAIN_OBJECTS:.o=.d)

NN_TARGET = $(BUILD_DIR)/nn/XNOR
SOLVER_TARGET = $(BUILD_DIR)/solver/solver
BLAS_TEST_TARGET = $(BUILD_DIR)/tests/blas_test
SOLVER_TEST_TARGET = $(BUILD_DIR)/tests/solver_test
NN_TEST_TARGET = $(BUILD_DIR)/tests/nn_test

GTK3_AVAILABLE := $(shell pkg-config --exists gtk+-3.0 2>/dev/null && echo "yes" || echo "no")

ifeq ($(GTK3_AVAILABLE),yes)
    GTK_CFLAGS = $(shell pkg-config --cflags gtk+-3.0)
    GTK_LIBS = $(shell pkg-config --libs gtk+-3.0)
else
    GTK_CFLAGS =
    GTK_LIBS =
endif

# Detect BLAS libraries
# Check for Nix environment first - use NIX_LDFLAGS if set
# Otherwise, try to detect available BLAS libraries
ifdef NIX_LDFLAGS
    # Using Nix environment - use provided library paths
    BLAS_LIBS = -L$(NIX_LDFLAGS) -lopenblas
else
    # Try to find Nix OpenBLAS
    NIX_BLAS_PATH := $(shell find /nix/store -name "libopenblas.so" 2>/dev/null | head -1 | xargs dirname 2>/dev/null || echo "")
    ifneq ($(NIX_BLAS_PATH),)
        BLAS_LIBS = -L$(NIX_BLAS_PATH) -lopenblas
    else
        # Try pkg-config
        BLAS_AVAILABLE := $(shell pkg-config --exists cblas 2>/dev/null && echo "yes" || echo "no")
        ifeq ($(BLAS_AVAILABLE),yes)
            BLAS_LIBS = $(shell pkg-config --libs cblas)
        else
            # Try system libraries
            # OpenBLAS includes both BLAS and CBLAS, so just -lopenblas should work
            # If that fails, try -lblas alone (some systems bundle CBLAS)
            BLAS_LIBS = -lopenblas
        endif
    endif
endif

all: dirs $(TARGET)

nn: dirs $(NN_TARGET)

solver: dirs $(SOLVER_TARGET)

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
	@mkdir -p $(BUILD_DIR)/image
	@mkdir -p $(BUILD_DIR)/detection
	@mkdir -p $(BUILD_DIR)/preprocessing
	@mkdir -p $(BUILD_DIR)/wordsearch
	@mkdir -p $(BUILD_DIR)/gui
	@mkdir -p $(BUILD_DIR)/nn
	@mkdir -p $(BUILD_DIR)/solver
	@mkdir -p $(BUILD_DIR)/tests

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

$(BLAS_TEST_TARGET): dirs $(BUILD_DIR)/tests/blas_test.o
	@echo "Linking BLAS test $@..."
	@echo "Using BLAS libraries: $(BLAS_LIBS)"
	$(CC) $(BUILD_DIR)/tests/blas_test.o -o $@ $(LDFLAGS) $(BLAS_LIBS)
	@echo "BLAS test build successful! Binary: $@"

test-blas: $(BLAS_TEST_TARGET)
	@echo "Running BLAS tests..."
	@$(BLAS_TEST_TARGET)

$(SOLVER_TEST_TARGET): dirs $(BUILD_DIR)/tests/solver_test.o $(BUILD_DIR)/solver/solver.o $(BUILD_DIR)/solver/search.o
	@echo "Linking solver test $@..."
	$(CC) $(BUILD_DIR)/tests/solver_test.o $(BUILD_DIR)/solver/solver.o $(BUILD_DIR)/solver/search.o -o $@ $(LDFLAGS)
	@echo "Solver test build successful! Binary: $@"

test-solver: $(SOLVER_TEST_TARGET)
	@echo "Running solver tests..."
	@$(SOLVER_TEST_TARGET)

$(NN_TEST_TARGET): dirs $(BUILD_DIR)/tests/nn_test.o
	@echo "Linking neural network test $@..."
	$(CC) $(BUILD_DIR)/tests/nn_test.o -o $@ $(LDFLAGS)
	@echo "Neural network test build successful! Binary: $@"

test-nn: $(NN_TEST_TARGET)
	@echo "Running neural network tests..."
	@$(NN_TEST_TARGET)

$(BUILD_DIR)/tests/%.o: $(SRC_DIR)/tests/%.c
	@echo "Compiling test $<..."
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -Isrc -MMD -MP -c $< -o $@

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
	@echo "  all           - Build the project (default)"
	@echo "  nn            - Build the XNOR neural network example"
	@echo "  solver        - Build the word search solver"
	@echo "  test-blas      - Build and run BLAS tests"
	@echo "  test-solver    - Build and run solver unit tests"
	@echo "  test-nn        - Build and run neural network tests"
	@echo "  clean         - Remove all build artifacts"
	@echo "  release       - Build optimized release"
	@echo "  install-deps  - Install GTK3 dependencies"
	@echo "  check-deps    - Check if dependencies are installed"
	@echo "  info          - Show build information"
	@echo ""
	@echo "Build directory: $(BUILD_DIR)"
	@echo "Source directory: $(SRC_DIR)"

.PHONY: all nn solver test-blas test-solver test-nn clean release install-deps check-deps dirs info help
