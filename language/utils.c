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
