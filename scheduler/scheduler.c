#include "scheduler.h"
#include "weights.c"

#define REALLY_LARGE_QUANTUM 1000000

/**
 * Round Robin scheduling algorithm
 *
 * \param queue         pointer to job queue
 * \param quantum       time slice for each job
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
			}
		}
	}
}

/**
 * Weighted Round Robin scheduling algorithm
 *
 * \param queue         pointer to job queue
 * \param quantum       max time slice for each job (each job will take a
 * 						certain % of quantum)
 * \param max_life_ms   maximum life of the scheduler
 */
void wrr_scheduler(JobQueue *queue, time_t quantum, time_t max_life_ms) {
	Job *job;
	time_t end_time = now() + max_life_ms;

	while ((job = next_job(queue)) != NULL || now() < end_time) {
		if (job != NULL) {
			double wq = get_operation_quantum(job->df, job->df, quantum);
			execute(job->df, job->state, (time_t)wq);

			// Remove completed jobs from the queue
			if (job->state->status == COMPLETED) {
				remove_job_from_queue(queue, job);
			}
		}
	}
}

/**
 * First In First Out scheduling algorithm
 *
 * \param queue         pointer to job queue
 * \param quantum       time slice for each job (does not matter for FIFO)
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
		}
	}
}
