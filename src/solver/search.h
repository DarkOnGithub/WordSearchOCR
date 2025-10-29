#pragma once

#include "solver.h"

Coord coord_res(int c, int word_size, int row, int column);

int int_res(int c, int word_size, int index);

int north_search(int row, int column, char *word, char **grid, int word_size);

Coord northeast_search(int row, int column, int columns, char *word, char **grid, int word_size);

int east_search(int row, int column, int columns, char *word, char **grid, int word_size);

Coord southeast_search(int row, int column, int rows, int columns, char *word, char **grid, int word_size);

int south_search(int row, int column, int rows, char *word, char **grid, int word_size);

Coord southwest_search(int row, int column, int rows, char *word, char **grid, int word_size);

int west_search(int row, int column, char *word, char **grid, int word_size);

Coord northweast_search(int row, int column, char *word, char **grid, int word_size);
