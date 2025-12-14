#include "language/language.h"
#include "logger/logger.h"
#include "scheduler/scheduler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
	// raise error if invalid number of args
	if (argc != 4) {
		fprintf(stderr,
				"Usage: %s <data_file.csv> <RR|WRR|FIFO> <output_log.csv>\n",
				argv[0]);
		exit(EXIT_FAILURE);
	}

	// determine scheduler algorithm
	int sch_algorithm = -1;
	if (strcmp(argv[2], "RR") == 0) {
		sch_algorithm = RR;
	} else if (strcmp(argv[2], "WRR") == 0) {
		sch_algorithm = WRR;
	} else if (strcmp(argv[2], "FIFO") == 0) {
		sch_algorithm = FIFO;
	}

	// raise error if invalid algorithm
	if (sch_algorithm == -1) {
		fprintf(stderr, "Invalid scheduler algorithm: %s\n", argv[2]);
		exit(EXIT_FAILURE);
	}

	// initialize executor with new dataframe
	Dataframe *df = malloc(sizeof(Dataframe));
	initialize(df, argv[1]);

	// initialize scheduler with scheduling algorithm & dataframe
	Scheduler *scheduler = malloc(sizeof(Scheduler));
	initialize_scheduler(scheduler, DEFAULT_QUANTUM_MS, sch_algorithm, df);

	// initialize logger
	log_init(argv[3]);

	char *raw_query = NULL;
	while (1) {
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
		// TODO: log all events inside scheduler
		schedule_query(scheduler, query);
		free(raw_query);
		raw_query = NULL;
	}

	// save logs to csv and cleanup
	scheduler->running = 0; // stop scheduler (so scheduler thread stops)
	cleanup_scheduler(scheduler);
	log_dump_csv();
	log_destroy();
}
