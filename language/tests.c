#include "headers/executor.h"

int main() {
    char queries[][50] = {
        "AVERAGE ( 2) ",
        "MEDIAN(  5   )",
        "INCREMENT(3,10)",
        "WRITE      ( 1 , 42 )",
        "WRITE_AT(0, 7, 3.14)",
        "COUNT(4, >=, 2.71)"
    };

    for (int i = 0; i < 6; i++) {
        Query query;
        parse(queries[i], &query);
        execute(&query);
    }

    return 0;
}