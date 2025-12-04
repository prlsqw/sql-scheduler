#include "headers/executor.h"
#include "headers/utils.h"

int main() {

    Dataframe df;
    initialize(&df, "data/test1.csv");

    // print dataframe info
    printf("Dataframe Info:\n");
    printf("  Rows: %d\n", df.num_rows);
    printf("  Columns: %d\n", df.num_cols);
    printf("  Cell Length: %d\n", df.cell_length);
    printf("  Header Length: %d\n", df.header_length);
    printf("  Row Width: %d\n\n", df.row_width);

    char queries[][50] = {
        "AVERAGE ( 0) ",
        "MEDIAN(  5   )",
        "INCREMENT(3,10)",
        "WRITE      ( 1 , 42 )",
        "WRITE_AT(0, 7, 3.14)",
        "COUNT(4, >=, 2.71)"
    };

    // test utility funcs
    char buffer[19];
    read_at(&df, 2, 3, buffer);
    printf("Value at (row 2, col 3): %s\n", buffer);
    write_at(&df, 2, 3, "147.33893178051888");
    printf("Wrote new value (147.33893178051888) at (row 2, col 3)\n");
    read_at(&df, 2, 3, buffer);
    printf("New value at (row 2, col 3): %s\n\n", buffer);


    for (int i = 0; i < 6; i++) {
        Query query;
        parse(queries[i], &query);
        execute(&df, &query);
    }

    cleanup(&df);
    return 0;
}