# ===================================================================
# WordSearchOCR - Modular Architecture Makefile
# ===================================================================

# Project configuration
PROJECT_NAME = wordsearch_ocr
BUILD_DIR = build
SRC_DIR = src
TARGET = $(BUILD_DIR)/$(PROJECT_NAME)

CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -g -I$(SRC_DIR)
LDFLAGS = -lm

MAIN_SOURCES = $(shell find $(SRC_DIR) -name "*.c" -type f -not -name "*test*.c" -not -name "*_test.c")
TEST_SOURCES = $(shell find $(SRC_DIR) -name "*test*.c" -o -name "*_test.c")
SOURCES = $(MAIN_SOURCES) $(TEST_SOURCES)
MAIN_OBJECTS = $(MAIN_SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
TEST_OBJECTS = $(TEST_SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
OBJECTS = $(MAIN_OBJECTS) $(TEST_OBJECTS)
DEPS = $(OBJECTS:.o=.d)

TEST_TARGET = $(BUILD_DIR)/tests/word_detection_test
NN_TARGET = $(BUILD_DIR)/nn/XNOR

GTK3_AVAILABLE := $(shell pkg-config --exists gtk+-3.0 2>/dev/null && echo "yes" || echo "no")

ifeq ($(GTK3_AVAILABLE),yes)
    GTK_CFLAGS = $(shell pkg-config --cflags gtk+-3.0)
    GTK_LIBS = $(shell pkg-config --libs gtk+-3.0)
else
    GTK_CFLAGS =
    GTK_LIBS =
endif

all: dirs $(TARGET)

test: check-deps dirs $(TEST_TARGET)

nn: dirs $(NN_TARGET)

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
	@mkdir -p $(BUILD_DIR)/tests

$(TARGET): $(MAIN_OBJECTS)
	@echo "Linking $@..."
	$(CC) $(MAIN_OBJECTS) -o $@ $(GTK_LIBS) $(LDFLAGS)
	@echo "Build successful! Binary: $@"

$(TEST_TARGET): $(TEST_OBJECTS) $(filter-out %/main.o, $(MAIN_OBJECTS))
	@echo "Linking test $@..."
	$(CC) $(TEST_OBJECTS) $(filter-out %/main.o, $(MAIN_OBJECTS)) -o $@ $(GTK_LIBS) $(LDFLAGS)
	@echo "Test build successful! Binary: $@"

$(NN_TARGET): $(BUILD_DIR)/nn/XNOR.o
	@echo "Linking neural network $@..."
	$(CC) $(BUILD_DIR)/nn/XNOR.o -o $@ $(LDFLAGS)
	@echo "Neural network build successful! Binary: $@"

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@echo "Compiling $<..."
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(GTK_CFLAGS) -MMD -MP -c $< -o $@

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

rebuild: clean all

run: $(TARGET)
	@echo "Running $(PROJECT_NAME)..."
	./$(TARGET)

run-test: $(TEST_TARGET)
	@echo "Running word detection test..."
	./$(TEST_TARGET)

run-nn: $(NN_TARGET)
	@echo "Running XNOR neural network..."
	./$(NN_TARGET)

debug: CFLAGS += -DDEBUG -O0
debug: all

release: CFLAGS += -O3 -DNDEBUG
release: all

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

help:
	@echo "=== $(PROJECT_NAME) Makefile Help ==="
	@echo ""
	@echo "Available targets:"
	@echo "  all           - Build the project (default)"
	@echo "  test          - Build the word detection test program"
	@echo "  nn            - Build the XNOR neural network example"
	@echo "  clean         - Remove all build artifacts"
	@echo "  rebuild       - Clean and rebuild everything"
	@echo "  run           - Build and run the main program"
	@echo "  run-test      - Build and run the word detection test"
	@echo "  run-nn        - Build and run the XNOR neural network"
	@echo "  debug         - Build with debug flags"
	@echo "  release       - Build optimized release"
	@echo "  install-deps  - Install GTK3 dependencies"
	@echo "  check-deps    - Check if dependencies are installed"
	@echo "  info          - Show build information"
	@echo ""
	@echo "Build directory: $(BUILD_DIR)"
	@echo "Source directory: $(SRC_DIR)"

.PHONY: all test nn clean rebuild run run-test run-nn debug release install-deps check-deps dirs info help
