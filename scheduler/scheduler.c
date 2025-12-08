#include "scheduler.h"
#include "algorithms.c"

void initialize_scheduler(Scheduler *scheduler, time_t quantum,
						  SchAlgorithm algorithm, Dataframe *df) {
	if (scheduler == NULL)
		return;

	scheduler->quantum = quantum;
	switch (algorithm) {
		case RR:
			scheduler->algorithm = rr_scheduler;
			break;
		case WRR:
			scheduler->algorithm = wrr_scheduler;
			break;
		case FIFO:
			scheduler->algorithm = fifo_scheduler;
			break;
		default:
			perror("Unknown scheduling algorithm");
			exit(EXIT_FAILURE);
	}

	scheduler->df = df;
	initialize_job_queue(&scheduler->queue);
}

int query_arrived(Scheduler *scheduler, Query *query) {
	if (scheduler == NULL || query == NULL) {
		return -1;
	}

	// create a job for the query
	Job *job = (Job *)malloc(sizeof(Job));
	job->df = scheduler->df;
	job->state = (ExecutionState *)malloc(sizeof(ExecutionState));
	job->state->query = query;
	job->state->status = CREATED;

	add_job_to_queue(&scheduler->queue, job);
	return 0;
}

void cleanup_scheduler(Scheduler *scheduler) {
	if (scheduler == NULL)
		return;
	cleanup_job_queue(&scheduler->queue);
}
