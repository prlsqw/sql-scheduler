#ifndef EXECUTOR_H
#define EXECUTOR_H 1

#include "parser.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef struct {
    // number of rows in the dataframe
    int num_rows;

    // number of columns in the dataframe
    int num_cols;

    // file descriptor for the opened CSV file
    FILE* file;

    // df enforces fixed-length cells
    int cell_length;

    // header row length in characters
    int header_length;

    // row width in characters (including commas and newline)
    int row_width;
} Dataframe;

typedef struct {
    // current row
    int curr_row_index;

    // current calculation
    double result;

    // rows counted so far
    int tally;
} Savestate_avg;

void initialize(Dataframe* df, const char* file_path);

void execute(Dataframe* df, Query* query);

void execute_average(Dataframe* df, int column_index);

void execute_median(Dataframe* df, int column_index);

void execute_increment(Dataframe* df, int column_index, double value);

void execute_write(Dataframe* df, int column_index, double value);

void execute_write_at(Dataframe* df, int column_index, int row_index, double value);

void execute_count(Dataframe* df, int column_index, int comparison_operator, double value);

void cleanup(Dataframe* df);

#endif
