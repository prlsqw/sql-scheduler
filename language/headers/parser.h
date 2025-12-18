#include "grammar.h"

int parse(char *command, Query *query);

void determine_operation(char *operation_str, Query *query);

void determine_column_index(char *column_name, Query *query);

void determine_comparison_operator(char *comp_str, Query *query);
