#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Split the given string at the first occurence of the given character
 * by replacing it with a null terminator. Throws an error if the character
 * is not found
 *
 * \param str      string to split
 * \param split_at character at which to split
 * \return         pointer to the character after the split character,
 */
char* split(char* str, char split_at) {
    int i = 0;
    while (str[i] != split_at && str[i] != '\0') {
        i++;
    }

    if (str[i] == '\0') {
        perror("Split Error: Character not found in string");
        exit(1);
    }

    str[i] = '\0';
    return str + i + 1;
}

/**
 * Remove leading whitespace from the given string by returning a pointer to
 * the first non-whitespace character
 *
 * \param str string to trim
 * \return    pointer to the first non-whitespace character
 */
char* ltrim(char* str) {
    int i = 0;
    while (str[i] == ' ' || str[i] == '\t' || str[i] == '\n') {
        i++;
    }
    return str + i;
}

/**
 * Remove trailing whitespace from the given string in place
 *
 * \param str string to trim
 */
void rtrim(char* str) {
    int i = strlen(str) - 1;
    while (i > 0 && (str[i] == ' ' || str[i] == '\t' || str[i] == '\n')) {
        i--;
    }
    str[i + 1] = '\0';
}

/**
 * Read the value between the column_index-th and (column_index + 1)-th comma
 * of the current line in the given file
 *
 * \param file         file pointer to read from
 * \param column_index index of the column to read
 * \param buffer       buffer to store the read value (should have at least 
 *                      df->cell_length capacity)
 */
void read_value_at_column(FILE* file, int column_index, char* buffer) {
    // find the column_index-th comma
    char ch = fgetc(file);
    while (ch != EOF && column_index > 0) {
        if (ch == ',') column_index--;
        ch = fgetc(file);
    }

    // read until the next comma or end of line
    int buf_index = 0;
    while (ch != EOF && ch != ',' && ch != '\n') {
        buffer[buf_index++] = ch;
        ch = fgetc(file);
    }

    buffer[buf_index] = '\0';
}

/**
 * Move the cursor of the file to right after \n or EOF, whichever comes first
 */
void next_line(FILE* file) {
    char ch = fgetc(file);
    while (ch != '\n' && ch != EOF) {
        ch = fgetc(file);
    }
}

/**
 * Compare two double values using the given comparison operator
 *
 * \param a   first value
 * \param op  comparison operator (<, <= as l, != as !, == as =, >= as g, >)
 * \param b   second value
 * \return    result of the comparison
 */
int compare(double a, char op, double b) {
    switch (op) {
        case '<':  return a < b;
        case 'l':  return a <= b; 
        case '!':  return a != b;
        case '=':  return a == b;
        case 'g':  return a >= b;
        case '>':  return a > b;
        default:
            perror("Comparison Error: Unknown comparison operator");
            exit(1);
    }
}