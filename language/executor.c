
#include <stdio.h>
#include <stdlib.h>

#include "headers/executor.h"
#include "headers/utils.h"

void initialize(Dataframe* df, const char* file_path) {
    df->file = fopen(file_path, "r+");
    if (df->file == NULL) {
        perror("Error opening file");
        exit(1);
    }

    df->num_rows = 1;
    df->num_cols = 1;
    df->cell_length = 0;
    df->header_length = 0;
    
    // num cols is the (number of ',' in line 1) + 1
    // header_length is the number of characters in line 1
    while (1) {
        char ch = fgetc(df->file);
        df->header_length++;

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

    df->row_width = (
        (df->num_cols * df->cell_length) // for each cell
        + df->num_cols - 1               // for commas
        + 1                              // for newline at the end
    );
}

void execute(Dataframe* df, Query* query, ExecutionState* state, time_t timeout) {
    if (df->file == NULL) {
        perror("Execute Error: Dataframe not initialized");
        exit(1);
    }

    switch (query->operation) {
        case AVERAGE:
            return execute_average(df, query->column_index, state, timeout);
        case MEDIAN:
            return execute_median(df, query->column_index, state, timeout);
        case INCREMENT:
            return execute_increment(
                df, query->column_index, query->arg1, state, timeout
            );
        case WRITE:
            return execute_write(
                df, query->column_index, query->arg1,
                state, timeout
            );
        case WRITE_AT:
            return execute_write_at(
                df, query->column_index, (int)query->arg1,
                query->arg2, state, timeout
            );
        case COUNT:
            return execute_count(
                df, query->column_index, (char)((int)query->arg1),
                query->arg2, state, timeout
            );
        default:
            perror("Execute Error: Unknown operation");
            exit(1);
    }
}

void execute_average(
    Dataframe* df, int column_index, ExecutionState* state, time_t timeout
) {

    time_t start_time = now();

    // housekeeping if call is the first one
    if (state->status == CREATED) {
        state->processed_rows = 0;
        state->tally = 0.0;
        state->status = INPROGRESS;

        // position after header row to mark where to start
        state->stream_position = df->header_length; 
    }
    
    // sanity check
    if (column_index < 0 || column_index >= df->num_cols) {
        perror("AVERAGE Error: Column index out of range");
        exit(1);
    }
    
    // seek to last position in file stream
    // Citation: https://man7.org/linux/man-pages/man3/fseek.3.html
    fseek(df->file, state->stream_position, SEEK_SET);

    // for each column, find the column_index-th comma
    // store the value after that comma in a buffer
    char buffer[df->cell_length + 1];

    // in each row, find the value at column_index
    while (
        (state->processed_rows < df->num_rows - 1)
         && (now() - start_time < timeout)
    ) {
        read_value_at_column(df->file, column_index, buffer);
        state->tally += atof(buffer);
        next_line(df->file);
        state->processed_rows++;
    }

    if (state->processed_rows == df->num_rows - 1) {
        state->status = COMPLETED;
    } else {
        // save position in file stream for next call
        state->stream_position = ftell(df->file);

        // swap context
        return;
    }

    printf("AVERAGE(%d): %f\n", column_index, state->tally / (df->num_rows - 1));
}

void execute_median(
    Dataframe* df, int column_index, ExecutionState* state, time_t timeout
) {
    printf("Executing MEDIAN on column %d\n", column_index);
    state->status = COMPLETED;
}

void execute_increment(
    Dataframe* df, int column_index, double value,
    ExecutionState* state, time_t timeout
) {
    printf("Executing INCREMENT on column %d by %f\n", column_index, value);
    state->status = COMPLETED;
}

void execute_write(
    Dataframe* df, int column_index, double value,
    ExecutionState* state, time_t timeout
) {
    printf("Executing WRITE on column %d with value %f\n", column_index, value);
    state->status = COMPLETED;
}

void execute_write_at(
    Dataframe* df, int column_index, int row_index, double value, ExecutionState* state, time_t timeout
) {
    printf("Executing WRITE_AT on column %d at row %d with value %f\n", column_index, row_index, value);
    state->status = COMPLETED;
}

void execute_count(
    Dataframe* df, int column_index, int comparison_operator, double value,
    ExecutionState* state, time_t timeout
) {

    time_t start_time = now();

    // housekeeping
    if (state->status == CREATED) {
        state->status = INPROGRESS;
        state->stream_position = df->header_length;
        state->processed_rows = 0;
        state->tally = 0;
    }

    if (column_index < 0 || column_index >= df->num_cols) {
        perror("COUNT Error: Column index out of range");
        exit(1);
    }
    
    // seek to last position in file stream
    fseek(df->file, state->stream_position, SEEK_SET);

    // for each column, find the column_index-th comma
    char buffer[df->cell_length + 1];

    // in each row, find the value at column_index
    while (
        (state->processed_rows < df->num_rows - 1)
         && (now() - start_time < timeout)
    ) {
        read_value_at_column(df->file, column_index, buffer);
        if (compare(atof(buffer), comparison_operator, value)) {
            state->tally += 1;
        }
        next_line(df->file);
        state->processed_rows++;
    }

    if (state->processed_rows == df->num_rows - 1) {
        state->status = COMPLETED;
    } else {
        // save position in file stream for next call
        state->stream_position = ftell(df->file);

        // swap context
        return;
    }

    printf("COUNT(%d, %c, %f): %d\n", column_index, (char)comparison_operator, value, (int)state->tally);
}

void cleanup(Dataframe* df) {
    if (df->file != NULL) {
        fclose(df->file);
        df->file = NULL;
    }
}
