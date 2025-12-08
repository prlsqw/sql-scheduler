
#include "headers/job_queue.h"

#include <string.h>

/**
 * Initialize the job queue
 *
 * \param queue  pointer to queue
 */
void initialize_job_queue(JobQueue *queue) {
	queue->size = 0;
	queue->capacity = INITIAL_JOB_QUEUE_CAPACITY;
	queue->iter = 0;
	queue->jobs = (Job **)malloc(queue->capacity * sizeof(Job *));
}

/**
 * Double the capacity of the current job queue
 *
 * \param queue  pointer to queue
 */
void double_job_queue_capacity(JobQueue *queue) {
	int new_capacity = queue->capacity << 1;
	Job **new_jobs = (Job **)malloc(new_capacity * sizeof(Job *));

	memcpy(new_jobs, queue->jobs, queue->size * sizeof(Job *));
	free(queue->jobs);

	queue->jobs = new_jobs;
	queue->capacity = new_capacity;
}

/**
 * Add a job to the queue
 *
 * \param queue  pointer to queue
 * \param job  pointer to the job to be added
 */
void add_job_to_queue(JobQueue *queue, Job *job) {
	if (queue->size >= queue->capacity) {
		double_query_queue_capacity(queue);
	}

	queue->jobs[queue->size] = job;
	queue->size++;
}

/**
 * Remove a job from the queue
 *
 * \param queue  pointer to queue
 * \param job  pointer to the job to be removed
 */
void remove_job_from_queue(JobQueue *queue, Job *job) {
	int i = 0;

	// find job position in the queue
	while (i < queue->size && queue->jobs[i] != job) {
		i++;
	}

	// if iterator was at the removed job, move it back one position
	if (queue->iter > i) {
		queue->iter--;
	}

	// shift remaining jobs left
	for (; i < queue->size - 1; i++) {
		queue->jobs[i] = queue->jobs[i + 1];
	}

	// decrease size
	queue->size--;

	// if iterator is now out of bounds, wrap it around
	queue->iter = queue->iter == queue->size ? 0 : queue->iter;
}

/**
 * A JobQueue iterator (in order of arrival time)
 *
 * \param queue  pointer to queue
 * \return       pointer to the next job, or NULL if the queue is empty
 */
Job *next_job(JobQueue *queue) {
	if (queue->size == 0) {
		return NULL;
	}

	Job *job = queue->jobs[queue->iter];
	queue->iter = (queue->iter + 1) % queue->size;
	return job;
}

/**
 * Free the memory allocated for the job queue
 *
 * \param queue  pointer to queue
 */
void cleanup_job_queue(JobQueue *queue) {
	// comment to ensure proper formatting
	free(queue->jobs);
}
