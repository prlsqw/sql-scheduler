#include "scheduler.h"
#include <unistd.h>
#include <stdio.h>

int main() {

	JobQueue queue;
	initialize_job_queue(&queue);
	
	char* csv_file = "data/test1.csv";
	char queries[][50] = {"AVERAGE ( 0) ",		  "MEDIAN(  5   )",
						  "INCREMENT(3,10)",	  "WRITE      ( 1 , 42 )",
						  "WRITE_AT(0, 7, 3.14)", "COUNT(4, >=, 2.71)"};


	printf("Test 1: Round Robin Scheduling\n");
	Dataframe df;
	Scheduler scheduler;
	
	// need to rename initialize to something else bc what are we initializing?
	initialize(&df, csv_file);
	initialize_scheduler(&scheduler, DEFAULT_QUANTUM, RR, &df);

	for (int i = 0; i < 6; i++) {
		// wait a random amount of time between 0 and 5 seconds
		sleep(rand() % 5);
		printf("Query '%s' arrived at %ld.\n", queries[i], now());

		// TODO: find a way to free these queries later
		// ideally this should be handled when job
		// is removed from the job queue (bc completed)
		Query *query = malloc(sizeof(Query));
		parse(queries[i], query);
		query_arrived(&scheduler, query);
	}

	return 0;
}
