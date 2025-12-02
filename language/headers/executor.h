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

} Dataframe;

void initialize(Dataframe* df, const char* file_path);

void execute(Dataframe* df, Query* query);

void execute_average(Dataframe* df, int column_index);

void execute_median(Dataframe* df, int column_index);

void execute_increment(Dataframe* df, int column_index, int value);

void execute_write(Dataframe* df, int column_index, int value);

void execute_write_at(Dataframe* df, int column_index, int row_index, double value);

void execute_count(Dataframe* df, int column_index, int comparison_operator, double value);

void cleanup(Dataframe* df);