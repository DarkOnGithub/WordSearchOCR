#pragma once

typedef struct
{
    int row;
    int column;

} Coord;

int solver(char **grid, char *word, int word_size, int first_letter, int rows, int columns);
