
#include "parser.h"

void execute(Query* query);

void execute_average(int column_index);

void execute_median(int column_index);

void execute_increment(int column_index, int value);

void execute_write(int column_index, int value);

void execute_write_at(int column_index, int row_index, double value);

void execute_count(int column_index, int comparison_operator, double value);