
#include <stdio.h>
#include <stdlib.h>
#include "headers/executor.h"

void initialize(Dataframe* df, const char* file_path) {
    df->file = fopen(file_path, "r");
    if (df->file == NULL) {
        perror("Error opening file");
        exit(1);
    }

    df->num_rows = 1;
    df->num_cols = 1;
    
    // num cols is the (number of ',' in line 1) + 1
    while (1) {
        char ch = fgetc(df->file);
        if (ch == '\n' || ch == EOF) break;
        if (ch == ',') df->num_cols++; 
    }

    // num rows is (number of '\n' in file) + 1
    while (1) {
        char ch = fgetc(df->file);
        if (ch == EOF) break;
        if (ch == '\n') df->num_rows++;
    }
}

void execute(Dataframe* df, Query* query) {
    if (df->file == NULL) {
        perror("Execute Error: Dataframe not initialized");
        exit(1);
    }

    switch (query->operation) {
        case AVERAGE:
            return execute_average(df, query->column_index);
        case MEDIAN:
            return execute_median(df, query->column_index);
        case INCREMENT:
            return execute_increment(df, query->column_index, query->arg1);
        case WRITE:
            return execute_write(df, query->column_index, query->arg1);
        case WRITE_AT:
            return execute_write_at(df, query->column_index, query->arg1, query->arg2);
        case COUNT:
            return execute_count(df, query->column_index, query->arg1, query->arg2);
        default:
            perror("Execute Error: Unknown operation");
            exit(1);
    }
}

void execute_average(Dataframe* df, int column_index) {
    printf("Executing AVERAGE on column %d\n", column_index);
}

void execute_median(Dataframe* df, int column_index) {
    printf("Executing MEDIAN on column %d\n", column_index);
}

void execute_increment(Dataframe* df, int column_index, int value) {
    printf("Executing INCREMENT on column %d by %d\n", column_index, value);
}

void execute_write(Dataframe* df, int column_index, int value) {
    printf("Executing WRITE on column %d with value %d\n", column_index, value);
}

void execute_write_at(Dataframe* df, int column_index, int row_index, double value) {
    printf("Executing WRITE_AT on column %d at row %d with value %f\n", column_index, row_index, value);
}

void execute_count(Dataframe* df, int column_index, int comparison_operator, double value) {
    printf("Executing COUNT on column %d with comparison operator %c and value %f\n", column_index, comparison_operator, value);
}

void cleanup(Dataframe* df) {
    if (df->file != NULL) {
        fclose(df->file);
        df->file = NULL;
    }
}