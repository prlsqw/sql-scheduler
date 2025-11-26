
#include <stdio.h>
#include <stdlib.h>
#include "headers/executor.h"

void execute(Query* query) {
    switch (query->operation) {
        case AVERAGE:
            return execute_average(query->column_index);
        case MEDIAN:
            return execute_median(query->column_index);
        case INCREMENT:
            return execute_increment(query->column_index, query->arg1);
        case WRITE:
            return execute_write(query->column_index, query->arg1);
        case WRITE_AT:
            return execute_write_at(query->column_index, query->arg1, query->arg2);
        case COUNT:
            return execute_count(query->column_index, query->arg1, query->arg2);
        default:
            perror("Execute Error: Unknown operation");
            exit(1);
    }
}

void execute_average(int column_index) {
    printf("Executing AVERAGE on column %d\n", column_index);
}

void execute_median(int column_index) {
    printf("Executing MEDIAN on column %d\n", column_index);
}

void execute_increment(int column_index, int value) {
    printf("Executing INCREMENT on column %d by %d\n", column_index, value);
}

void execute_write(int column_index, int value) {
    printf("Executing WRITE on column %d with value %d\n", column_index, value);
}

void execute_write_at(int column_index, int row_index, double value) {
    printf("Executing WRITE_AT on column %d at row %d with value %f\n", column_index, row_index, value);
}

void execute_count(int column_index, int comparison_operator, double value) {
    printf("Executing COUNT on column %d with comparison operator %c and value %f\n", column_index, comparison_operator, value);
}