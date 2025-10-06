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

# SDL2 detection and configuration
SDL2_AVAILABLE := $(shell pkg-config --exists sdl2 2>/dev/null && echo "yes" || echo "no")
SDL2_IMAGE_AVAILABLE := $(shell pkg-config --exists SDL2_image 2>/dev/null && echo "yes" || echo "no")

# Set SDL flags if available
ifeq ($(SDL2_AVAILABLE),yes)
    SDL_CFLAGS = $(shell pkg-config --cflags sdl2)
    SDL_LIBS = $(shell pkg-config --libs sdl2)
    ifeq ($(SDL2_IMAGE_AVAILABLE),yes)
        SDL_CFLAGS += $(shell pkg-config --cflags SDL2_image)
        SDL_LIBS += $(shell pkg-config --libs SDL2_image)
    endif
else
    SDL_CFLAGS =
    SDL_LIBS =
endif

# Default target
all: check-deps dirs $(TARGET)

# Test target
test: check-deps dirs $(TEST_TARGET)

# Check dependencies
check-deps:
	@if [ "$(SDL2_AVAILABLE)" != "yes" ] || [ "$(SDL2_IMAGE_AVAILABLE)" != "yes" ]; then \
		echo "Error: Missing required dependencies."; \
		echo "Required: libsdl2-dev libsdl2-image-dev"; \
		echo "Run 'make install-deps' to install them."; \
		exit 1; \
	fi
	@echo "Dependencies OK: SDL2=$(SDL2_AVAILABLE), SDL2_image=$(SDL2_IMAGE_AVAILABLE)"

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
	$(CC) $(MAIN_OBJECTS) -o $@ $(SDL_LIBS) $(LDFLAGS)
	@echo "Build successful! Binary: $@"

$(TEST_TARGET): $(TEST_OBJECTS) $(filter-out %/main.o, $(MAIN_OBJECTS))
	@echo "Linking test $@..."
	$(CC) $(TEST_OBJECTS) $(filter-out %/main.o, $(MAIN_OBJECTS)) -o $@ $(SDL_LIBS) $(LDFLAGS)
	@echo "Test build successful! Binary: $@"

# Compilation with dependency generation
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@echo "Compiling $<..."
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(SDL_CFLAGS) -MMD -MP -c $< -o $@

# Include generated dependencies
-include $(DEPS)

# Installation targets
install-deps:
	@echo "Installing SDL2 and SDL2_image development libraries..."
	@command -v apt-get >/dev/null 2>&1 && { \
		sudo apt-get update && \
		sudo apt-get install -y libsdl2-dev libsdl2-image-dev; \
	} || { \
		echo "Error: apt-get not found."; \
		echo "Please install SDL2 manually."; \
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
	@echo "SDL2: $(SDL2_AVAILABLE)"
	@echo "SDL2_image: $(SDL2_IMAGE_AVAILABLE)"
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
	@echo "  install-deps  - Install SDL2 dependencies"
	@echo "  check-deps    - Check if dependencies are installed"
	@echo "  info          - Show build information"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Build directory: $(BUILD_DIR)"
	@echo "Source directory: $(SRC_DIR)"

# Phony targets
.PHONY: all test clean rebuild run run-test debug release install-deps check-deps dirs info help
