#include "headers/executor.h"

int main() {

    Dataframe df;
    initialize(&df, "data/test1.csv");

    char queries[][50] = {
        "AVERAGE ( 0) ",
        "MEDIAN(  5   )",
        "INCREMENT(3,10)",
        "WRITE      ( 1 , 42 )",
        "WRITE_AT(0, 7, 3.14)",
        "COUNT(4, >=, 2.71)"
    };

    for (int i = 0; i < 6; i++) {
        Query query;
        parse(queries[i], &query);
        execute(&df, &query);
    }

    cleanup(&df);
    return 0;
}