/*
The accepted query syntaxes are as follows (from the proposal).
	AVERAGE(column_x)
	MEDIAN(column_y)
	INCREMENT(column_y, 1)
	WRITE(column_z, 100)
	WRITE_AT(column_z, row_indx, 10)
	COUNT(column_x, >, 10)
*/

/*
Let COL be a column identifier, INT be an integer, and FLT be a floating-point
number. Let CMP = < | <= | ! | = | >= | >.

The grammar of this DSL is defined as:
	QUERY := AVERAGE   (COL)
		   | MEDIAN    (COL)
		   | INCREMENT (COL, FLT)
		   | WRITE     (COL, FLT)
		   | WRITE_AT  (COL, INT, FLT)
		   | COUNT     (COL, CMP, FLT)
*/

// the following operations are allowed, none implies parse error
#ifndef GRAMMAR_H
#define GRAMMAR_H 1

static const char* ComparisonOps[] = { "<", "<=", "!", "=", ">=", ">" };

typedef enum {
	AVERAGE,
	MEDIAN,
	INCREMENT,
	WRITE,
	WRITE_AT,
	COUNT,
	NONE
} Operation;

static const char* QueryOps[] = {"AVERAGE", "MEDIAN", "INCREMENT", "WRITE", "WRITE_AT", "COUNT"};

// post-parse query structure
typedef struct {
	// each query has an operation type
	Operation operation;

	// first argument of every operation is column name
	// parser needs to convert it to column index
	int column_index;

	// second argument is either:
	// - -1 (to indicate query does not have this arg)
	// - integer value (for WRITE_AT)
	// - char value (for COUNT)
	// - float value (for INCREMENT, WRITE)
	// so, can be stored as a double and interpreted accordingly
	double arg1;

	// third argument is either:
	// - -1.0
	// - float (for WRITE_AT, COUNT)
	double arg2;
} Query;

// maximum length of a double
// Citation: https://stackoverflow.com/a/1701272
#define MAX_DOUBLE_LENGTH 1080

#endif
