#include "solver.h"
#include "search.h"
#include <stdio.h>
#include <stdlib.h>

int solver(char **grid, char *word, int word_size, int first_letter, int rows, int columns) //search the word in the grid
{
	for (int row = 0; row < rows; row++)
	{
		for (int column = 0; column < columns; column++)
		{
			if (grid[row][column] == first_letter) //if finds the first letter of the word in the grid
			{
				int int_pos = north_search(row - 1, column, word, grid, word_size);
				if (int_pos != -1)
				{
					printf("(%i,%i)(%i,%i)\n", column, row, column, int_pos);
					return EXIT_SUCCESS;
				}
				int_pos = south_search(row + 1, column, rows, word, grid, word_size);
				if (int_pos != -1)
				{
					printf("(%i,%i)(%i,%i)\n", column, row, column, int_pos);
					return EXIT_SUCCESS;
				}
				int_pos = west_search(row, column - 1, word, grid, word_size);
				if (int_pos != -1)
				{
					printf("(%i,%i)(%i,%i)\n", column, row, int_pos, row);
					return EXIT_SUCCESS;
				}
				int_pos = east_search(row, column + 1, columns, word, grid, word_size);
				if (int_pos != -1)
				{
					printf("(%i,%i)(%i,%i)\n", column, row, int_pos, row);
					return EXIT_SUCCESS;
				}
				Coord pos;
				pos = northeast_search(row - 1, column + 1, columns, word, grid, word_size);
				if (pos.row != -1)
				{
					printf("(%i,%i)(%i,%i)\n", column, row, pos.column, pos.row);
                                        return EXIT_SUCCESS;
				}
				pos = southeast_search(row + 1, column + 1, rows, columns, word, grid, word_size);
				if (pos.row != -1)
				{
					printf("(%i,%i)(%i,%i)\n", column, row, pos.column, pos.row);
                                        return EXIT_SUCCESS;
				}
				pos = southwest_search(row + 1, column - 1, rows, word, grid, word_size);
				if (pos.row != -1)
				{
					printf("(%i,%i)(%i,%i)\n", column, row, pos.column, pos.row);
                                        return EXIT_SUCCESS;
				}
				pos = northweast_search(row - 1, column - 1, word, grid, word_size);
				if (pos.row != -1)
				{
					printf("(%i,%i)(%i,%i)\n", column, row, pos.column, pos.row);
                                        return EXIT_SUCCESS;
				}
			}
		}
	}
	printf("Not Found\n");
	return EXIT_SUCCESS;	
}
