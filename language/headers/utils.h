#include "executor.h"
#include <sys/time.h>

#define ARRAY_LEN(array) sizeof(array) / sizeof(array[0])

// ino_t is uint64_t
#define MAXIMUM_INODE_CHARACTERS 21

/**
 * Split the given string at the first occurence of the given character
 * by replacing it with a null terminator. Throws an error if the character
 * is not found
 *
 * \param str_p      pointer to string to split
 * \param split_at character at which to split
 * \return         -1 if error, 0 if success. Modifies str on success.
 */
int split(char **str_p, char split_at);

/**
 * Remove leading whitespace from the given string by returning a pointer to
 * the first non-whitespace character
 *
 * \param str string to trim
 * \return    pointer to the first non-whitespace character
 */
char *ltrim(char *str);

/**
 * Remove trailing whitespace from the given string in place
 *
 * \param str string to trim
 */
void rtrim(char *str);

/**
 * Read the value between the column_index-th and (column_index + 1)-th comma of
 * the current line in the given file
 *
 * \param file         file pointer to read from
 * \param column_index index of the column to read
 * \param buffer       buffer to store the read value
 */
void read_value_at_column(FILE *file, int column_index, char *buffer);

/**
 * Move the cursor of the file to right after \n or EOF, whichever comes first
 */
void next_line(FILE *file);

/**
 * compare two double values using the given comparison operator
 *
 * \param a   first value
 * \param op  comparison operator
 * \param b   second value
 * \return    result of the comparison
 */
int compare(double a, char op, double b);

/**
 * Move the cursor of the dataframe file to the row-th row
 * and col-th column in dataframe
 *
 * \param df      dataframe whose file cursor to move
 * \param row     row index to move to (0-indexed)
 * \param col     column index to move to (0-indexed)
 */
void move_to(Dataframe *df, int row, int col);

/**
 * Read the value at the given (row, col) position in the file
 *
 * \param df      dataframe whose file pointer to read from
 * \param row     row index to read from (0-indexed)
 * \param col     column index to read from (0-indexed)
 * \param buffer  buffer to store the read value (should have enough space
 *                 i.e, determined by df->cell_length)
 */
void read_at(Dataframe *df, int row, int col, char *buffer);

/**
 * Write the value at the given (row, col) position in the file
 *
 * \param df     dataframe whose file pointer to read from
 * \param row    row index to read from (0-indexed)
 * \param col    column index to read from (0-indexed)
 * \param value  value to write (should be df->cell_length sized)
 */
void write_at(Dataframe *df, int row, int col, char *value);

/**
 * Get current time in seconds since epoch
 *
 * \return current time in seconds since epoch
 */
time_t now();

/**
 * Insert the given value into the sorted array using binary search
 *
 * \param values     array to insert into
 * \param count      number of elements in the array
 * \param capacity   capacity of the array
 * \param value      value to insert
 */
void insert_sorted(double **values, int *count, int *capacity, double value);

/**
 * Align value for final cell to write with padding or truncation dependent on
 * cell_length
 *
 * \param value         double that needs to be truncated or padded into
 * cell_length
 * \param cell          cell that stores our final output string which we will
 * write with
 * \param cell_length   length of cells in csv based on df struct
 */
void align_num(double value, char *cell, int cell_length);
