/**
 * Split the given string at the first occurence of the given character
 * by replacing it with a null terminator. Throws an error if the character
 * is not found
 *
 * \param str      string to split
 * \param split_at character at which to split
 * \return         pointer to the character after the split character,
 */
char* split(char* str, char split_at);

/**
 * Remove leading whitespace from the given string by returning a pointer to
 * the first non-whitespace character
 *
 * \param str string to trim
 * \return    pointer to the first non-whitespace character
 */
char* ltrim(char* str);


/**
 * Remove trailing whitespace from the given string in place
 *
 * \param str string to trim
 */
void rtrim(char* str);


/**
 * Read the value between the column_index-th and (column_index + 1)-th comma of
 * the current line in the given file
 *
 * \param file         file pointer to read from
 * \param column_index index of the column to read
 * \param buffer       buffer to store the read value
 */
void read_value_at_column(FILE* file, int column_index, char* buffer);


/**
 * Move the cursor of the file to right after \n or EOF, whichever comes first
 */
void next_line(FILE* file);