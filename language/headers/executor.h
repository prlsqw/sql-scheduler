#ifndef EXECUTOR_H
#define EXECUTOR_H 1

#include "parser.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>

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
    // user query is a part of execution state
    Query* query;

    // to keep track of how much work was done previously
    int processed_rows;

    // intermediate result storage
    double tally;

    // execution status
    enum { CREATED, INPROGRESS, COMPLETED } status;

    // position in file stream
    long stream_position;

    // flag to keep track of whether user has called write_at
    int user_write_at;
} ExecutionState;

void initialize(Dataframe* df, const char* file_path);

void execute(Dataframe* df, ExecutionState* state, time_t timeout);

void execute_average(Dataframe* df, ExecutionState* state, time_t timeout);

void execute_median(Dataframe* df, ExecutionState* state, time_t timeout);

void execute_increment(Dataframe* df, ExecutionState* state, time_t timeout);

void execute_write(Dataframe* df, ExecutionState* state, time_t timeout);

void execute_write_at(Dataframe* df, ExecutionState* state, time_t timeout);

void execute_count(Dataframe* df, ExecutionState* state, time_t timeout);

void cleanup(Dataframe* df);

#endif
