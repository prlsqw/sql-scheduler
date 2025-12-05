
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

void execute(Dataframe* df, ExecutionState* state, time_t timeout) {
    if (df->file == NULL) {
        perror("Execute Error: Dataframe not initialized");
        exit(1);
    }

    switch (state->query->operation) {
        case AVERAGE:
            return execute_average(df, state, timeout);
        case MEDIAN:
            return execute_median(df, state, timeout);
        case INCREMENT:
            return execute_increment(df, state, timeout);
        case WRITE:
            return execute_write(df, state, timeout);
        case WRITE_AT:
            return execute_write_at(df, state, timeout);
        case COUNT:
            return execute_count(df, state, timeout);
        default:
            perror("Execute Error: Unknown operation");
            exit(1);
    }
}

void execute_average(Dataframe* df, ExecutionState* state, time_t timeout) {

    int column_index = state->query->column_index;
    time_t start_time = now();

    // housekeeping if call is the first one
    if (state->status == CREATED) {
        state->processed_rows = 0;
        state->tally = 0.0;
        state->status = INPROGRESS;

        // position after header row to mark where to start
        state->stream_position = df->header_length; 

        // sanity check
        if (column_index < 0 || column_index >= df->num_cols) {
            perror("AVERAGE Error: Column index out of range");
            exit(1);
        }
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

void execute_median(Dataframe* df, ExecutionState* state, time_t timeout) {
    int column_index = state->query->column_index;
    printf("Executing MEDIAN on column %d\n", column_index);
    state->status = COMPLETED;
}

void execute_increment(Dataframe* df, ExecutionState* state, time_t timeout) {
    int column_index = state->query->column_index;
    double value = state->query->arg1;
    time_t start_time = now();

    // housekeeping if call is the first one
    if (state->status == CREATED) {
        state->processed_rows = 0;
        state->status = INPROGRESS;

        // sanity check
        if (column_index < 0 || column_index >= df->num_cols) {
            perror("Increment Error: Column index out of range");
            exit(1);
        }
    }
    char buffer[df->cell_length + 1];
    
    while (
        (state->processed_rows < df->num_rows - 1)
         && (now() - start_time < timeout)
    ) {    
        // 1) Read current value at (row, col).
        read_at(df, state->processed_rows, column_index, buffer);

        // 2) Convert to double and increment.
        double current = atof(buffer);
        double updated = current + value;

        state->query->arg1 = state->processed_rows;
        state->query->arg2 = updated;
        state->user_write_at = 0;
        // 3) Write the updated value back. (Write at handles truncation and padding)
        execute_write_at(df, state, timeout);
        state->processed_rows++;
    }
    if (state->processed_rows == df->num_rows - 1) {
        state->status = COMPLETED;
    } else {
        state->query->arg1 = value;
        // swap context
        return;
    }
    printf("INCREMENT(%d, %f)\n", column_index, value);
}

void execute_write(Dataframe* df, ExecutionState* state, time_t timeout) {
    int column_index = state->query->column_index;
    double value = state->query->arg1;
    time_t start_time = now();

    // housekeeping if call is the first one
    if (state->status == CREATED) {
        state->processed_rows = 0;
        state->status = INPROGRESS;

        // sanity check
        if (column_index < 0 || column_index >= df->num_cols) {
            perror("WRITE Error: Column index out of range");
            exit(1);
        }
    }

     while (
        (state->processed_rows < df->num_rows - 1)
         && (now() - start_time < timeout)
    ) {
        state->query->arg1 = state->processed_rows;
        state->query->arg2 = value;
        state->user_write_at = 0;
        execute_write_at(df, state, timeout);
        state->processed_rows++;
    }
    if (state->processed_rows == df->num_rows - 1) {
        state->status = COMPLETED;
    } else {
        state->query->arg1 = value;
        // swap context
        return;
    }
    printf("WRITE( %d, %f)\n", column_index, value);
    state->status = COMPLETED;
}

void execute_write_at(Dataframe* df, ExecutionState* state, time_t timeout) {
    int column_index = state->query->column_index;
    int row_index = (int)state->query->arg1;
    double value = state->query->arg2;

    // housekeeping if call is the first one
    if (state->status == CREATED) {
        state->status = INPROGRESS;

        // sanity check
        if (column_index < 0 || column_index >= df->num_cols) {
            perror("WRITE_AT Error: Column index out of range");
            exit(1);
        }
        if (row_index < 0 || row_index >= df->num_rows) {
            perror("WRITE_AT Error: Row index out of range");
            exit(1);
        }
    }

    char cell[df->cell_length + 1];
    
    align_num(value, cell, df->cell_length);
    
    // Actually write at (row_index, column_index).
    write_at(df, row_index, column_index, cell);

    if(state->user_write_at == 1){
    printf("WRITE_AT(%d, %d, %f)\n", column_index, row_index, value);
    state->status = COMPLETED;
    }
}

void execute_count(Dataframe* df, ExecutionState* state, time_t timeout) {
    
    int column_index = state->query->column_index;
    int comparison_operator = (int)state->query->arg1;
    double value = state->query->arg2;
    
    time_t start_time = now();
    
    // housekeeping
    if (state->status == CREATED) {
        state->status = INPROGRESS;
        state->stream_position = df->header_length;
        state->processed_rows = 0;
        state->tally = 0;

        if (column_index < 0 || column_index >= df->num_cols) {
            perror("COUNT Error: Column index out of range");
            exit(1);
        }
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
