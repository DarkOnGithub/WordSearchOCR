# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -g
TARGET = wordsearch
SRCDIR = src
SOURCES = $(wildcard $(SRCDIR)/*.c) $(wildcard $(SRCDIR)/**/*.c)
OBJECTS = $(SOURCES:.c=.o)

SDL2_AVAILABLE := $(shell pkg-config --exists sdl2 2>/dev/null && echo "yes" || echo "no")
SDL2_IMAGE_AVAILABLE := $(shell pkg-config --exists SDL2_image 2>/dev/null && echo "yes" || echo "no")

# Installation targets
install-deps:
	@echo "Installing SDL2 and SDL2_image development libraries..."
	@command -v apt-get >/dev/null 2>&1 && { \
		sudo apt-get update && \
		sudo apt-get install -y libsdl2-dev libsdl2-image-dev; \
	} || { \
		echo "Error: apt-get not found. Please install SDL2 manually:"; \
		exit 1; \
	}

check-deps:
	@if [ "$(SDL2_AVAILABLE)" != "yes" ] || [ "$(SDL2_IMAGE_AVAILABLE)" != "yes" ]; then \
		echo "Missing required dependencies. Run 'make install-deps' to install them."; \
		echo "Required: libsdl2-dev libsdl2-image-dev"; \
		exit 1; \
	fi

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

# Build target
all: check-deps $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(SDL_LIBS)

%.o: %.c
	$(CC) $(CFLAGS) $(SDL_CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Rebuild everything
rebuild: clean all

# Help target
help:
	@echo "Available targets:"
	@echo "  all           - Build the project (default)"
	@echo "  clean         - Remove build artifacts"
	@echo "  rebuild       - Clean and rebuild the project"
	@echo "  install-deps  - Install SDL2 and SDL2_image development libraries"
	@echo "  check-deps    - Check if required dependencies are installed"
	@echo "  help          - Show this help message"

.PHONY: all clean rebuild install-deps check-deps help
