# ===================================================================
# WordSearchOCR - Modular Architecture Makefile
# ===================================================================

# Project configuration
PROJECT_NAME = wordsearch
BUILD_DIR = build
SRC_DIR = src
TARGET = $(BUILD_DIR)/$(PROJECT_NAME)

# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -g -I$(SRC_DIR)
LDFLAGS = -lm  # Math library for atan2f, sqrtf, expf, etc.

# Source files discovery (exclude test files from main build)
MAIN_SOURCES = $(shell find $(SRC_DIR) -name "*.c" -type f -not -name "*test*.c" -not -name "*_test.c")
TEST_SOURCES = $(shell find $(SRC_DIR) -name "*test*.c" -o -name "*_test.c")
SOURCES = $(MAIN_SOURCES) $(TEST_SOURCES)
MAIN_OBJECTS = $(MAIN_SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
TEST_OBJECTS = $(TEST_SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
OBJECTS = $(MAIN_OBJECTS) $(TEST_OBJECTS)
DEPS = $(OBJECTS:.o=.d)

# Test targets
TEST_TARGET = $(BUILD_DIR)/word_detection_test

# GTK3 detection and configuration
GTK3_AVAILABLE := $(shell pkg-config --exists gtk+-3.0 2>/dev/null && echo "yes" || echo "no")

# Set GTK flags if available
ifeq ($(GTK3_AVAILABLE),yes)
    GTK_CFLAGS = $(shell pkg-config --cflags gtk+-3.0)
    GTK_LIBS = $(shell pkg-config --libs gtk+-3.0)
else
    GTK_CFLAGS =
    GTK_LIBS =
endif

# Default target
all: check-deps dirs $(TARGET)

# Test target
test: check-deps dirs $(TEST_TARGET)

# Check dependencies
check-deps:
	@if [ "$(GTK3_AVAILABLE)" != "yes" ]; then \
		echo "Error: Missing required dependencies."; \
		echo "Required: libgtk-3-dev"; \
		echo "Run 'make install-deps' to install them."; \
		exit 1; \
	fi
	@echo "Dependencies OK: GTK3=$(GTK3_AVAILABLE)"

# Create build directories
dirs:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BUILD_DIR)/core/image
	@mkdir -p $(BUILD_DIR)/processing
	@mkdir -p $(BUILD_DIR)/grid
	@mkdir -p $(BUILD_DIR)/ocr

# Linking
$(TARGET): $(MAIN_OBJECTS)
	@echo "Linking $@..."
	$(CC) $(MAIN_OBJECTS) -o $@ $(GTK_LIBS) $(LDFLAGS)
	@echo "Build successful! Binary: $@"

$(TEST_TARGET): $(TEST_OBJECTS) $(filter-out %/main.o, $(MAIN_OBJECTS))
	@echo "Linking test $@..."
	$(CC) $(TEST_OBJECTS) $(filter-out %/main.o, $(MAIN_OBJECTS)) -o $@ $(GTK_LIBS) $(LDFLAGS)
	@echo "Test build successful! Binary: $@"

# Compilation with dependency generation
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@echo "Compiling $<..."
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(GTK_CFLAGS) -MMD -MP -c $< -o $@

# Include generated dependencies
-include $(DEPS)

# Installation targets
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

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)

# Clean and rebuild
rebuild: clean all

# Run the program (if it exists)
run: $(TARGET)
	@echo "Running $(PROJECT_NAME)..."
	./$(TARGET)

# Run the test program
run-test: $(TEST_TARGET)
	@echo "Running word detection test..."
	./$(TEST_TARGET)

# Debug build
debug: CFLAGS += -DDEBUG -O0
debug: all

# Release build
release: CFLAGS += -O3 -DNDEBUG
release: all

# Show build information
info:
	@echo "=== Build Information ==="
	@echo "Project: $(PROJECT_NAME)"
	@echo "Target: $(TARGET)"
	@echo "Sources: $(words $(SOURCES))"
	@echo "Objects: $(words $(OBJECTS))"
	@echo "GTK3: $(GTK3_AVAILABLE)"
	@echo "Compiler: $(CC)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "LDFLAGS: $(LDFLAGS)"

# Help target
help:
	@echo "=== $(PROJECT_NAME) Makefile Help ==="
	@echo ""
	@echo "Available targets:"
	@echo "  all           - Build the project (default)"
	@echo "  test          - Build the word detection test program"
	@echo "  clean         - Remove all build artifacts"
	@echo "  rebuild       - Clean and rebuild everything"
	@echo "  run           - Build and run the main program"
	@echo "  run-test      - Build and run the word detection test"
	@echo "  debug         - Build with debug flags"
	@echo "  release       - Build optimized release"
	@echo "  install-deps  - Install GTK3 dependencies"
	@echo "  check-deps    - Check if dependencies are installed"
	@echo "  info          - Show build information"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Build directory: $(BUILD_DIR)"
	@echo "Source directory: $(SRC_DIR)"

# Phony targets
.PHONY: all test clean rebuild run run-test debug release install-deps check-deps dirs info help
