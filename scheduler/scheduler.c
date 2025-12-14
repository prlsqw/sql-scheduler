#include "scheduler.h"
#include "../logger/logger.h"

#define REALLY_LARGE_QUANTUM 1000000

/**
 * Round Robin scheduling algorithm
 *
 * \param scheduler     pointer to scheduler object
 * \param quantum       time slice for each job
 * \param max_life_ms   maximum life of the scheduler
 */
void rr_scheduler(Scheduler *scheduler, time_t quantum, time_t max_life_ms) {
	Job *job;
	time_t end_time = now() + max_life_ms;

	// Run until max_life_ms is reached and all queries are processed
	while ((job = next_job(&scheduler->queue)) != NULL ||
		   (scheduler->running && now() < end_time)) {
		if (job != NULL) {
			log_start(job->id);
			execute(job->df, job->state, quantum);

			// Remove completed jobs from the queue
			if (job->state->status == COMPLETED) {
				log_stop(job->id, true);
				remove_job_from_queue(&scheduler->queue, job);
			} else {
				log_stop(job->id, false);
			}
		}
	}
}

/**
 * Weighted Round Robin scheduling algorithm
 *
 * \param scheduler     pointer to scheduler object
 * \param quantum       max time slice for each job (each job will take a
 * 						certain % of quantum)
 * \param max_life_ms   maximum life of the scheduler
 */
void wrr_scheduler(Scheduler *scheduler, time_t quantum, time_t max_life_ms) {
	Job *job;
	time_t end_time = now() + max_life_ms;

	while ((job = next_job(&scheduler->queue)) != NULL ||
		   (scheduler->running && now() < end_time)) {
		if (job != NULL) {
			double wq =
				get_operation_quantum(job->df, job->state->query, quantum);

			log_start(job->id);
			execute(job->df, job->state, (time_t)wq);

			// Remove completed jobs from the queue
			if (job->state->status == COMPLETED) {
				log_stop(job->id, true);
				remove_job_from_queue(&scheduler->queue, job);
			} else {
				log_stop(job->id, false);
			}
		}
	}
}

/**
 * First In First Out scheduling algorithm
 *
 * \param scheduler     pointer to scheduler object
 * \param quantum       time slice for each job (does not matter for FIFO)
 * \param max_life_ms   maximum life of the scheduler
 */
void fifo_scheduler(Scheduler *scheduler, time_t quantum, time_t max_life_ms) {
	Job *job;
	time_t end_time = now() + max_life_ms;

	while ((job = next_job(&scheduler->queue)) != NULL ||
		   (scheduler->running && now() < end_time)) {
		if (job != NULL) {
			// execute job to completion
			log_start(job->id);
			while (job->state->status != COMPLETED) {
				execute(job->df, job->state, REALLY_LARGE_QUANTUM);
			}

			// since job is done,
			log_stop(job->id, true);
			remove_job_from_queue(&scheduler->queue, job);
		}
	}
}

/**
 * Worker Function for Scheduling Scheduler
 * \param arg      Pointer to scheduler
 */
void *scheduler_thread_main(void *arg) {
	Scheduler *scheduler = (Scheduler *)arg;

	// Run the chosen algorithm until it finishes
	scheduler->algorithm(scheduler, scheduler->quantum_ms,
						 scheduler->max_life_ms);

	return NULL;
}
