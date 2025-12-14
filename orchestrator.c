#include <stdio.h>
#include <stdlib.h>

#include "language/language.h"
#include "scheduler/scheduler.h"

// temp change for draft pr
int main(int argc, char **argv) {

	if (argc != 3) {
		fprintf(stderr, "Usage: %s <data_file.csv> <RR|WRR|FIFO>\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	// New Jersey: only look at first char & default to WRR
	int sch_algorithm = WRR;
	if (argv[2][0] == 'R') {
		sch_algorithm = RR;
	} else if (argv[2][0] == 'F') {
		sch_algorithm = FIFO;
	}

	Dataframe *df;
	initialize(df, argv[1]);

	// New Jersey: use default quantum and max life for now
	Scheduler *scheduler;
	initialize_scheduler(scheduler, DEFAULT_QUANTUM_MS, sch_algorithm, df);

	char *raw_query = NULL;
	while (1) {
		// prompt user for input
		printf(">>> ");

		size_t len = 0;
		ssize_t read = getline(&raw_query, &len, stdin);

		if (read == -1 || strcmp(raw_query, ":quit\n") == 0) {
			free(raw_query);
			break;
		}

		// process the query
		Query *query = malloc(sizeof(Query));
		parse(raw_query, query);

		// send to scheduler & hope that it deals with freeing query
		schedule_query(scheduler, query);
		free(raw_query);
	}
}
