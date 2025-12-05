// parse the grammar of the language into an AST of sorts

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "headers/parser.h"
#include "headers/utils.h"

void parse(char *command, Query *query) {
	query->operation = NONE;
	query->column_index = -1;
	query->arg1 = -1;
	query->arg2 = -1.0;

	// sanitize command by removing leading whitespace
	char *operation = ltrim(command);

	// determine operation by splitting at '('
	char *column_name = ltrim(split(operation, '('));

	// remove trailing whitespace from operation
	rtrim(operation);

	// determine operation type
	determine_operation(operation, query);
	if (query->operation == NONE) {
		perror("Parse Error: Unknown operation");
		exit(1);
	}

	// if operation is AVERAGE or MEDIAN, column_name is terminated at ')'
	if (query->operation == AVERAGE || query->operation == MEDIAN) {
		split(column_name, ')');
		rtrim(column_name);

		determine_column_index(column_name, query);
	} else {
		// column_name is terminated at ','
		char *arg1_str = ltrim(split(column_name, ','));
		rtrim(column_name);

		if (query->operation == INCREMENT || query->operation == WRITE) {
			// arg1_str is terminated at ')'
			split(arg1_str, ')');
			rtrim(arg1_str);

			determine_column_index(column_name, query);
			query->arg1 = atof(arg1_str);
			return;
		}

		// arg1_str is terminated at ','
		char *arg2_str = ltrim(split(arg1_str, ','));
		rtrim(arg1_str);

		// arg2_str is terminated at ')'
		split(arg2_str, ')');
		rtrim(arg2_str);

		determine_column_index(column_name, query);

		if (query->operation == COUNT) {
			// arg1_str is a comparison operator, store as char
			determine_comparison_operator(arg1_str, query);
		} else {
			// arg1_str is an integer (row index for WRITE_AT)
			query->arg1 = atoi(arg1_str);
		}
		query->arg2 = atof(arg2_str);
	}
}

void determine_operation(char *operation_str, Query *query) {
	if (strcmp(operation_str, "AVERAGE") == 0) {
		query->operation = AVERAGE;
	} else if (strcmp(operation_str, "MEDIAN") == 0) {
		query->operation = MEDIAN;
	} else if (strcmp(operation_str, "INCREMENT") == 0) {
		query->operation = INCREMENT;
	} else if (strcmp(operation_str, "COUNT") == 0) {
		query->operation = COUNT;
	} else if (strcmp(operation_str, "WRITE") == 0) {
		query->operation = WRITE;
	} else if (strcmp(operation_str, "WRITE_AT") == 0) {
		query->operation = WRITE_AT;
	}
}

void determine_column_index(char *column_name, Query *query) {
	query->column_index = atoi(column_name);
}

void determine_comparison_operator(char *comp_str, Query *query) {
	if (strcmp(comp_str, "<") == 0) {
		query->arg1 = '<';
	} else if (strcmp(comp_str, "<=") == 0) {
		query->arg1 = 'l';
	} else if (strcmp(comp_str, "!") == 0) {
		query->arg1 = '!';
	} else if (strcmp(comp_str, "=") == 0) {
		query->arg1 = '=';
	} else if (strcmp(comp_str, ">=") == 0) {
		query->arg1 = 'g';
	} else if (strcmp(comp_str, ">") == 0) {
		query->arg1 = '>';
	} else {
		perror("Parse Error: Unknown comparison operator");
		exit(1);
	}
}
