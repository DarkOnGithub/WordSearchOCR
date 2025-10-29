#include "search.h"
#include "solver.h"

Coord coord_res(int c, int word_size, int row, int column)
{
	Coord res;
	if (c == word_size)
	{
    		res.row = row;
    		res.column = column;
		return res;
	}
	else
	{
		res.row = -1;
		res.column = -1;
		return res;
	}
}

int int_res(int c, int word_size, int index)
{
	if (c == word_size)
	{
		return index;
	}
	else
	{
		return -1;
	}
}

int north_search(int row, int column, char *word, char **grid, int word_size)
{
	int c = 1;
	while ((row >= 0) && (c < word_size) && (grid[row][column]) == word[c])
	{
		c++;
		row--;
	}
	return int_res(c, word_size, row + 1);
}

Coord northeast_search(int row, int column, int columns, char *word, char **grid, int word_size)
{
	int c = 1;
	while ((row >= 0) && (column < columns) &&(c < word_size) && (grid[row][column]) == word[c])
	{
		c++;
		row--;
		column++;
	}
	return coord_res(c, word_size, row + 1, column - 1);
}

int east_search(int row, int column, int columns, char *word, char **grid, int word_size)
{
	int c = 1;
	while ((column < columns) && (c < word_size) && (grid[row][column]) == word[c])
	{
		c++;
		column++;
	}
	return int_res(c, word_size, column - 1);
}

Coord southeast_search(int row, int column, int rows, int columns, char *word, char **grid, int word_size)
{
	int c = 1;
	while ((column < columns) && (row < rows) && (c < word_size) && (grid[row][column]) == word[c])
	{
		c++;
		row++;
		column++;
	}
	return coord_res(c, word_size, row - 1, column - 1);
}

int south_search(int row, int column, int rows, char *word, char **grid, int word_size)
{
	int c = 1;
	while ((row < rows) && (c < word_size) && (grid[row][column]) == word[c])
	{
		c++;
		row++;
	}
	return int_res(c, word_size, row - 1);
}

Coord southwest_search(int row, int column, int rows, char *word, char **grid, int word_size)
{
	int c = 1;
	while ((column >= 0) && (row < rows) && (c < word_size) && (grid[row][column]) == word[c])
	{
		c++;
		row++;
		column--;
	}
	return coord_res(c, word_size, row - 1, column + 1);
}

int west_search(int row, int column, char *word, char **grid, int word_size)
{
	int c = 1;
	while ((column >= 0) && (c < word_size) && (grid[row][column]) == word[c])
	{
		c++;
		column--;
	}
	return int_res(c, word_size, column + 1);
}

Coord northweast_search(int row, int column, char *word, char **grid, int word_size)
{
	int c = 1;
	while ((column >= 0) && (row >= 0) && (c < word_size) && (grid[row][column]) == word[c])
	{
		c++;
		row--;
		column--;
	}
	return coord_res(c, word_size, row + 1, column + 1);
}
