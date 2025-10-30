#include "../solver/solver.h"
#include "../solver/search.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdarg.h>
#include <ctype.h>

#define GREEN "\033[32m"
#define RED "\033[31m"
#define RESET "\033[0m"

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

void test_start(const char *test_name) {
    printf("Testing: %s... ", test_name);
    tests_run++;
}

void test_pass(void) {
    printf(GREEN "PASS" RESET "\n");
    tests_passed++;
}

void test_fail(const char *reason) {
    printf(RED "FAIL" RESET " - %s\n", reason);
    tests_failed++;
}

void test_summary(void) {
    printf("\n=== Solver Test Summary ===\n");
    printf("Total tests: %d\n", tests_run);
    printf(GREEN "Passed: %d" RESET "\n", tests_passed);
    printf(RED "Failed: %d" RESET "\n", tests_failed);
    printf("Success rate: %.1f%%\n", tests_run > 0 ? (100.0 * tests_passed / tests_run) : 0.0);
}

char **create_grid(const char *rows[], int num_rows, int num_cols) {
    char **grid = malloc(num_rows * sizeof(char *));
    for (int i = 0; i < num_rows; i++) {
        grid[i] = malloc(num_cols * sizeof(char));
        for (int j = 0; j < num_cols; j++) {
            grid[i][j] = rows[i][j];
        }
    }
    return grid;
}

void free_grid(char **grid, int rows) {
    for (int i = 0; i < rows; i++) {
        free(grid[i]);
    }
    free(grid);
}

bool test_east_search(void) {
    test_start("Word found going EAST");

    const char *grid_data[] = {
        "HELLO",
        "WORLD",
        "TEST "
    };
    char **grid = create_grid(grid_data, 3, 5);
    char word[] = "HELLO";
    int result = solver(grid, word, 5, 'H', 3, 5);

    free_grid(grid, 3);

    if (result == EXIT_SUCCESS) {
        test_pass();
        return true;
    } else {
        test_fail("EAST search failed");
        return false;
    }
}

bool test_south_search(void) {
    test_start("Word found going SOUTH");

    const char *grid_data[] = {
        "HELLO",
        "E    ",
        "L    ",
        "L    ",
        "O    "
    };
    char **grid = create_grid(grid_data, 5, 5);
    char word[] = "HELLO";
    int result = solver(grid, word, 5, 'H', 5, 5);

    free_grid(grid, 5);

    if (result == EXIT_SUCCESS) {
        test_pass();
        return true;
    } else {
        test_fail("SOUTH search failed");
        return false;
    }
}

bool test_west_search(void) {
    test_start("Word found going WEST");

    const char *grid_data[] = {
        "OLLEH",
        "     ",
        "TEST "
    };
    char **grid = create_grid(grid_data, 3, 5);
    char word[] = "HELLO";
    int result = solver(grid, word, 5, 'H', 3, 5);

    free_grid(grid, 3);

    if (result == EXIT_SUCCESS) {
        test_pass();
        return true;
    } else {
        test_fail("WEST search failed");
        return false;
    }
}

bool test_north_search(void) {
    test_start("Word found going NORTH");

    const char *grid_data[] = {
        "O    ",
        "L    ",
        "L    ",
        "E    ",
        "H    "
    };
    char **grid = create_grid(grid_data, 5, 5);
    char word[] = "HELLO";
    int result = solver(grid, word, 5, 'H', 5, 5);

    free_grid(grid, 5);

    if (result == EXIT_SUCCESS) {
        test_pass();
        return true;
    } else {
        test_fail("NORTH search failed");
        return false;
    }
}

bool test_northeast_search(void) {
    test_start("Word found going NORTHEAST");

    const char *grid_data[] = {
        "    O",
        "   L ",
        "  L  ",
        " E   ",
        "H    "
    };
    char **grid = create_grid(grid_data, 5, 5);
    char word[] = "HELLO";
    int result = solver(grid, word, 5, 'H', 5, 5);

    free_grid(grid, 5);

    if (result == EXIT_SUCCESS) {
        test_pass();
        return true;
    } else {
        test_fail("NORTHEAST search failed");
        return false;
    }
}

bool test_southeast_search(void) {
    test_start("Word found going SOUTHEAST");

    const char *grid_data[] = {
        "H    ",
        " E   ",
        "  L  ",
        "   L ",
        "    O"
    };
    char **grid = create_grid(grid_data, 5, 5);
    char word[] = "HELLO";
    int result = solver(grid, word, 5, 'H', 5, 5);

    free_grid(grid, 5);

    if (result == EXIT_SUCCESS) {
        test_pass();
        return true;
    } else {
        test_fail("SOUTHEAST search failed");
        return false;
    }
}

bool test_southwest_search(void) {
    test_start("Word found going SOUTHWEST");

    const char *grid_data[] = {
        "    H",
        "   E ",
        "  L  ",
        " L   ",
        "O    "
    };
    char **grid = create_grid(grid_data, 5, 5);
    char word[] = "HELLO";
    int result = solver(grid, word, 5, 'H', 5, 5);

    free_grid(grid, 5);

    if (result == EXIT_SUCCESS) {
        test_pass();
        return true;
    } else {
        test_fail("SOUTHWEST search failed");
        return false;
    }
}

bool test_northwest_search(void) {
    test_start("Word found going NORTHWEST");

    const char *grid_data[] = {
        "O    ",
        " L   ",
        "  L  ",
        "   E ",
        "    H"
    };
    char **grid = create_grid(grid_data, 5, 5);
    char word[] = "HELLO";
    int result = solver(grid, word, 5, 'H', 5, 5);

    free_grid(grid, 5);

    if (result == EXIT_SUCCESS) {
        test_pass();
        return true;
    } else {
        test_fail("NORTHWEST search failed");
        return false;
    }
}

bool test_word_not_found(void) {
    test_start("Word not found in grid");

    const char *grid_data[] = {
        "ABCDE",
        "FGHIJ",
        "KLMNO"
    };
    char **grid = create_grid(grid_data, 3, 5);
    char word[] = "HELLO";
    int result = solver(grid, word, 5, 'H', 3, 5);

    free_grid(grid, 3);

    if (result == EXIT_SUCCESS) {
        test_pass();
        return true;
    } else {
        test_fail("Expected EXIT_SUCCESS even when word not found");
        return false;
    }
}

bool test_single_letter(void) {
    test_start("Single letter word");

    const char *grid_data[] = {
        "ABC",
        "DEF",
        "GHI"
    };
    char **grid = create_grid(grid_data, 3, 3);
    char word[] = "A";
    int result = solver(grid, word, 1, 'A', 3, 3);

    free_grid(grid, 3);

    if (result == EXIT_SUCCESS) {
        test_pass();
        return true;
    } else {
        test_fail("Single letter search failed");
        return false;
    }
}

bool test_boundary_search(void) {
    test_start("Word at grid boundaries");

    const char *grid_data[] = {
        "HELLO",
        "     ",
        "     "
    };
    char **grid = create_grid(grid_data, 3, 5);
    char word[] = "HELLO";
    int result = solver(grid, word, 5, 'H', 3, 5);

    free_grid(grid, 3);

    if (result == EXIT_SUCCESS) {
        test_pass();
        return true;
    } else {
        test_fail("Boundary search failed");
        return false;
    }
}

bool test_multiple_occurrences(void) {
    test_start("Multiple occurrences (finds first)");

    const char *grid_data[] = {
        "HELLO",
        "HELLO",
        "WORLD"
    };
    char **grid = create_grid(grid_data, 3, 5);
    char word[] = "HELLO";
    int result = solver(grid, word, 5, 'H', 3, 5);

    free_grid(grid, 3);

    if (result == EXIT_SUCCESS) {
        test_pass();
        return true;
    } else {
        test_fail("Multiple occurrences test failed");
        return false;
    }
}

bool test_small_grid(void) {
    test_start("Small grid (2x2)");

    const char *grid_data[] = {
        "HI",
        "AB"
    };
    char **grid = create_grid(grid_data, 2, 2);
    char word[] = "HI";
    int result = solver(grid, word, 2, 'H', 2, 2);

    free_grid(grid, 2);

    if (result == EXIT_SUCCESS) {
        test_pass();
        return true;
    } else {
        test_fail("Small grid test failed");
        return false;
    }
}

bool test_case_sensitive(void) {
    test_start("Case sensitive search");

    const char *grid_data[] = {
        "HELLO",
        "WORLD",
        "TEST "
    };
    char **grid = create_grid(grid_data, 3, 5);
    char word[] = "hello";
    int result = solver(grid, word, 5, 'h', 3, 5);

    free_grid(grid, 3);

    if (result == EXIT_SUCCESS) {
        test_pass();
        return true;
    } else {
        test_fail("Case sensitive test failed");
        return false;
    }
}

char **load_grid_from_file(const char *filename, int *rows, int *cols) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        return NULL;
    }

    char buffer[256];
    if (!fgets(buffer, sizeof(buffer), file)) {
        fclose(file);
        return NULL;
    }

    *cols = 0;
    while (buffer[*cols] != '\n' && buffer[*cols] != '\0' && buffer[*cols] != '\r') {
        (*cols)++;
    }

    if (*cols == 0) {
        fclose(file);
        return NULL;
    }

    rewind(file);
    char **grid = NULL;
    *rows = 0;

    while (fgets(buffer, sizeof(buffer), file)) {
        int len = strlen(buffer);
        if (len > 0 && buffer[len-1] == '\n') {
            buffer[len-1] = '\0';
            len--;
        }
        if (len > 0 && buffer[len-1] == '\r') {
            buffer[len-1] = '\0';
            len--;
        }

        if (len != *cols) {
            for (int i = 0; i < *rows; i++) {
                free(grid[i]);
            }
            free(grid);
            fclose(file);
            return NULL;
        }

        grid = realloc(grid, (*rows + 1) * sizeof(char *));
        grid[*rows] = malloc((*cols + 1) * sizeof(char));

        for (int i = 0; i < *cols; i++) {
            grid[*rows][i] = toupper(buffer[i]);
        }
        grid[*rows][*cols] = '\0';

        (*rows)++;
    }

    fclose(file);
    return grid;
}

bool test_grid1_horizontal(void) {
    test_start("grid_1.txt - HORIZONTAL word");

    int rows, cols;
    char **grid = load_grid_from_file("grids/grid_1.txt", &rows, &cols);

    if (!grid) {
        test_fail("Failed to load grid_1.txt");
        return false;
    }

    char word[] = "HORIZONTAL";
    int result = solver(grid, word, 10, 'H', rows, cols);

    free_grid(grid, rows);

    if (result == EXIT_SUCCESS) {
        test_pass();
        return true;
    } else {
        test_fail("Failed to find HORIZONTAL in grid_1");
        return false;
    }
}

bool test_grid1_hello(void) {
    test_start("grid_1.txt - HELLO word (backwards)");

    int rows, cols;
    char **grid = load_grid_from_file("grids/grid_1.txt", &rows, &cols);

    if (!grid) {
        test_fail("Failed to load grid_1.txt");
        return false;
    }

    char word[] = "HELLO";
    int result = solver(grid, word, 5, 'H', rows, cols);

    free_grid(grid, rows);

    if (result == EXIT_SUCCESS) {
        test_pass();
        return true;
    } else {
        test_fail("Failed to find HELLO in grid_1");
        return false;
    }
}

bool test_grid2_hello(void) {
    test_start("grid_2.txt - HELLO word");

    int rows, cols;
    char **grid = load_grid_from_file("grids/grid_2.txt", &rows, &cols);

    if (!grid) {
        test_fail("Failed to load grid_2.txt");
        return false;
    }

    char word[] = "HELLO";
    int result = solver(grid, word, 5, 'H', rows, cols);

    free_grid(grid, rows);

    if (result == EXIT_SUCCESS) {
        test_pass();
        return true;
    } else {
        test_fail("Failed to find HELLO in grid_2");
        return false;
    }
}

bool test_grid2_world(void) {
    test_start("grid_2.txt - WORLD word");

    int rows, cols;
    char **grid = load_grid_from_file("grids/grid_2.txt", &rows, &cols);

    if (!grid) {
        test_fail("Failed to load grid_2.txt");
        return false;
    }

    char word[] = "WORLD";
    int result = solver(grid, word, 5, 'W', rows, cols);

    free_grid(grid, rows);

    if (result == EXIT_SUCCESS) {
        test_pass();
        return true;
    } else {
        test_fail("Failed to find WORLD in grid_2");
        return false;
    }
}

bool test_grid2_python(void) {
    test_start("grid_2.txt - PYTHON word");

    int rows, cols;
    char **grid = load_grid_from_file("grids/grid_2.txt", &rows, &cols);

    if (!grid) {
        test_fail("Failed to load grid_2.txt");
        return false;
    }

    char word[] = "PYTHON";
    int result = solver(grid, word, 6, 'P', rows, cols);

    free_grid(grid, rows);

    if (result == EXIT_SUCCESS) {
        test_pass();
        return true;
    } else {
        test_fail("Failed to find PYTHON in grid_2");
        return false;
    }
}

bool test_grid2_code(void) {
    test_start("grid_2.txt - CODE word");

    int rows, cols;
    char **grid = load_grid_from_file("grids/grid_2.txt", &rows, &cols);

    if (!grid) {
        test_fail("Failed to load grid_2.txt");
        return false;
    }

    char word[] = "CODE";
    int result = solver(grid, word, 4, 'C', rows, cols);

    free_grid(grid, rows);

    if (result == EXIT_SUCCESS) {
        test_pass();
        return true;
    } else {
        test_fail("Failed to find CODE in grid_2");
        return false;
    }
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    printf("=== Solver Unit Test Suite ===\n\n");

    test_east_search();
    test_south_search();
    test_west_search();
    test_north_search();
    test_northeast_search();
    test_southeast_search();
    test_southwest_search();
    test_northwest_search();
    test_word_not_found();
    test_single_letter();
    test_boundary_search();
    test_multiple_occurrences();
    test_small_grid();
    test_case_sensitive();

    printf("\n=== Tests using grid files ===\n");
    test_grid1_horizontal();
    test_grid1_hello();
    test_grid2_hello();
    test_grid2_world();
    test_grid2_python();
    test_grid2_code();

    test_summary();

    return (tests_failed > 0) ? 1 : 0;
}

