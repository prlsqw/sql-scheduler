#include "scheduler.h"

void initialize_scheduler(Scheduler *scheduler, time_t quantum,
						  SchAlgorithm algorithm, Dataframe *df) {
	scheduler->quantum_ms = quantum;
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

	// TODO: start the actual scheduler algorithm in a new thread
	// use pthreads and make sure the actual scheduler runs independent
	// of this main thread.
	int rc = pthread_create(&scheduler->thread, NULL, scheduler_thread_main,
							scheduler);
	if (rc != 0) {
		perror("Failed to create scheduler thread");
		exit(EXIT_FAILURE);
	}
}

int schedule_query(Scheduler *scheduler, Query *query) {
	if (scheduler == NULL || query == NULL) {
		perror("Either scheduler or query is NULL in schedule_query");
		exit(EXIT_FAILURE);
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
	if (scheduler == NULL) {
		perror("Scheduler is NULL before cleanup");
		exit(EXIT_FAILURE);
	}
	// Wait for scheduler thread to finish
	pthread_join(scheduler->thread, NULL);
	cleanup_job_queue(&scheduler->queue);
}
