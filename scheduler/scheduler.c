#include "scheduler.h"

#define REALLY_LARGE_QUANTUM 1000000

/**
 * Round Robin scheduling algorithm
 *
 * \param queue      pointer to job queue
 * \param quantum    time slice for each job
 * \param max_life_ms   maximum life of the scheduler
 */
void rr_scheduler(JobQueue *queue, time_t quantum, time_t max_life_ms) {
	Job *job;
	time_t end_time = now() + max_life_ms;

	// Run until max_life_ms is reached and all queries are processed
	while ((job = next_job(queue)) != NULL || now() < end_time) {
		if (job != NULL) {
			execute(job->df, job->state, quantum);

			// Remove completed jobs from the queue
			if (job->state->status == COMPLETED) {
				remove_job_from_queue(queue, job);
				free(job->state);
				free(job);
			}
		}
	}
}

/**
 * Weighted Round Robin scheduling algorithm
 *
 * \param queue      pointer to job queue
 * \param quantum    max time slice for each job (each job will take a
 * 						certain % of quantum)
 * \param max_life_ms   maximum life of the scheduler
 */
void wrr_scheduler(JobQueue *queue, time_t quantum, time_t max_life_ms) {
	Job *job;
	time_t end_time = now() + max_life_ms;

	while ((job = next_job(queue)) != NULL || now() < end_time) {
		// ?? need to implement; right now this does not work.
	}
}

/**
 * First In First Out scheduling algorithm
 *
 * \param queue      pointer to job queue
 * \param quantum    time slice for each job (does not matter for FIFO)
 * \param max_life_ms   maximum life of the scheduler
 */
void fifo_scheduler(JobQueue *queue, time_t quantum, time_t max_life_ms) {
	Job *job;
	time_t end_time = now() + max_life_ms;

	while ((job = next_job(queue)) != NULL || now() < end_time) {
		if (job != NULL) {
			// execute job to completion
			while (job->state->status != COMPLETED) {
				execute(job->df, job->state, REALLY_LARGE_QUANTUM);
			}

			// since job is done,
			remove_job_from_queue(queue, job);
			// TODO: freeing jobs and job states should be handled by
			// the job queue when jobs are removed
			free(job->state);
			free(job);
		}
	}
}

/**
 * Worker Function for Scheduling Scheduler
 * \param arg      Pointer to scheduler
 */
static void *scheduler_thread_main(void *arg) {
    Scheduler *scheduler = (Scheduler *)arg;

    // Run the chosen algorithm until it finishes
    scheduler->algorithm(&scheduler->queue,
                         scheduler->quantum_ms,
                         scheduler->max_life_ms);

    return NULL;
}
