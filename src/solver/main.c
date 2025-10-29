#include "solver.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <err.h>

int main(int argc, char *argv[])
{

	if (argc != 3) //checks the number of arguments
	{
		errx(EXIT_FAILURE, "The fonction requires 2 arguments");
	}

	FILE *gride_file = fopen(argv[1], "r"); //opens the file

	if (gride_file == NULL)
	{
		errx(EXIT_FAILURE, "Unable to open the file.");
	}

	int columns = 0;
	int c;

	while (((c = fgetc(gride_file)) != EOF) && (c != '\n')) //counts the number of elements (columns) in the first line
	{
		columns++;
	}

	if (columns < 5) //the number of columns must be 5 at least
	{
		errx(EXIT_FAILURE, "Not enough columns in the grid");
	}

	rewind(gride_file); //comeback at the beginning of the file

	char **grid = NULL;
	int rows = 0;
	char buffer[columns+1];
	int i = 0;

	while ((c = fgetc(gride_file)) != EOF) //reads the file and create the grid
	{
		if (c == '\n')
		{
			buffer[i] = '\0';

			if (i != columns)
				errx(EXIT_FAILURE, "Invalid line length");

			grid = realloc(grid, (rows + 1) * sizeof(char *));
			if (!grid)
				errx(EXIT_FAILURE, "Memory allocation failed");

			grid[rows] = malloc((columns + 1) * sizeof(char));
			if (!grid[rows])
				errx(EXIT_FAILURE, "Memory allocation failed");

			strcpy(grid[rows], buffer);
			rows++;
			i = 0;
		}
		else
		{
			if (i < columns)
			{
				if ((c >= 'a') && (c <= 'z'))
				{
					buffer[i] = c + ('A' - 'a');
				}
				else if ((c >= 'A') && (c <= 'Z'))
				{
					buffer[i] = c;
				}
				else
				{
					errx(EXIT_FAILURE, "Not a letter");
				}
				i++;
			}
			else
			{
				errx(EXIT_FAILURE, "Too many characters in a line");
			}
		}
	}

	if (i > 0) //Handles the last line if the file does not end with '\n'
	{
		buffer[i] = '\0';
		if (i != columns)
			errx(EXIT_FAILURE, "Invalid line length (last line)");
		grid = realloc(grid, (rows + 1) * sizeof(char *));
		if (!grid)
			errx(EXIT_FAILURE, "Memory allocation failed");
		grid[rows] = malloc((columns + 1) * sizeof(char));
		if (!grid[rows])
			errx(EXIT_FAILURE, "Memory allocation failed");
		strcpy(grid[rows], buffer);
		rows++;
	}


	fclose(gride_file);

	if (rows < 5) //the number of rows must be 5 at least
        {
                errx(EXIT_FAILURE, "Not enough rows");
        }

	int word_size = 0;
	while ((argv[2][word_size]) != 0) //counts the size of the word and puts the letters in capital letters
	{
		char c = argv[2][word_size];

		if ((c >= 'a') && (c <= 'z'))
		{
			argv[2][word_size] = c + ('A' - 'a');
		}
		else if (!((c >= 'A') && (c <= 'Z')))
		{
			errx(EXIT_FAILURE, "Not a letter");
		}
		word_size++;
	}

	if (word_size <= 0) //checks if there's a word
        {
                errx(EXIT_FAILURE, "The word is empty");
        }

	char first_letter = argv[2][0];

	return solver(grid, argv[2], word_size, first_letter, rows, columns); //finds the postion of the word in the grid

}
