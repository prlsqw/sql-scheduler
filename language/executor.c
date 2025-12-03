
#include <stdio.h>
#include <stdlib.h>

#include "headers/executor.h"
#include "headers/utils.h"

void initialize(Dataframe* df, const char* file_path) {
    df->file = fopen(file_path, "r");
    if (df->file == NULL) {
        perror("Error opening file");
        exit(1);
    }

    df->num_rows = 1;
    df->num_cols = 1;
    df->cell_length = 0;
    
    // num cols is the (number of ',' in line 1) + 1
    while (1) {
        char ch = fgetc(df->file);
        if (ch == '\n' || ch == EOF) break;
        if (ch == ',') df->num_cols++; 
    }

    // cell_length is the number of characters in the first cell
    while (1) {
        char ch = fgetc(df->file);
        if (ch == ',' || ch == '\n' || ch == EOF) break;
        df->cell_length++;
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
            return execute_write_at(df, query->column_index, (int)query->arg1, query->arg2);
        case COUNT:
            return execute_count(df, query->column_index, (char)((int)query->arg1), query->arg2);
        default:
            perror("Execute Error: Unknown operation");
            exit(1);
    }
}

void execute_average(Dataframe* df, int column_index) {
    // sanity check
    if (column_index < 0 || column_index >= df->num_cols) {
        perror("AVERAGE Error: Column index out of range");
        exit(1);
    }
    
    // seek to beginning of file
    // Citation: https://man7.org/linux/man-pages/man3/fseek.3.html
    fseek(df->file, 0, SEEK_SET);
    
    // ignore the header row
    next_line(df->file);

    // for each column, find the column_index-th comma
    // store the value after that comma in a buffer
    char buffer[df->cell_length + 1];
    double result = 0.0;

    // in each row, find the value at column_index
    for (int i = 0; i < df->num_rows - 1; i++) {
        read_value_at_column(df->file, column_index, buffer);
        result += atof(buffer);
        next_line(df->file);
    }

    printf("AVERAGE(%d): %f\n", column_index, result / (df->num_rows - 1));
}

void execute_median(Dataframe* df, int column_index) {
    printf("Executing MEDIAN on column %d\n", column_index);
}

void execute_increment(Dataframe* df, int column_index, double value) {
    printf("Executing INCREMENT on column %d by %f\n", column_index, value);
}

void execute_write(Dataframe* df, int column_index, double value) {
    printf("Executing WRITE on column %d with value %f\n", column_index, value);
}

void execute_write_at(Dataframe* df, int column_index, int row_index, double value) {
    printf("Executing WRITE_AT on column %d at row %d with value %f\n", column_index, row_index, value);
}

void execute_count(Dataframe* df, int column_index, int comparison_operator, double value) {
    if (column_index < 0 || column_index >= df->num_cols) {
        perror("COUNT Error: Column index out of range");
        exit(1);
    }
    
    // seek to beginning of file
    fseek(df->file, 0, SEEK_SET);
    
    // ignore the header row
    next_line(df->file);

    // for each column, find the column_index-th comma
    char buffer[df->cell_length + 1];
    int result = 0;

    // in each row, find the value at column_index
    for (int i = 0; i < df->num_rows - 1; i++) {
        read_value_at_column(df->file, column_index, buffer);
        result += compare(atof(buffer), comparison_operator, value);
        next_line(df->file);
    }

    printf("COUNT(%d, %c, %f): %d\n", column_index, (char)comparison_operator, value, result);
}

void cleanup(Dataframe* df) {
    if (df->file != NULL) {
        fclose(df->file);
        df->file = NULL;
    }
}