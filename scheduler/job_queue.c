
#include "scheduler.h"

#include <string.h>

// TODO: add, remove, double should be thread safe

/**
 * Initialize the job queue
 *
 * \param queue  pointer to queue
 */
void initialize_job_queue(JobQueue *queue) {
	queue->size = 0;
	queue->head = NULL;
    queue->tail = NULL;
    queue->curr = NULL;
}

// /**
//  * Double the capacity of the current job queue
//  *
//  * \param queue  pointer to queue
//  */
// void double_job_queue_capacity(JobQueue *queue) {
// 	int new_capacity = queue->capacity << 1;
// 	Job **new_jobs = (Job **)malloc(new_capacity * sizeof(Job *));

// 	memcpy(new_jobs, queue->jobs, queue->size * sizeof(Job *));
// 	free(queue->jobs);

// 	queue->jobs = new_jobs;
// 	queue->capacity = new_capacity;
// }

/**
 * Add a job to the queue
 *
 * \param queue  pointer to queue
 * \param job  pointer to the job to be added
 */
void add_job_to_queue(JobQueue *queue, Job *job) {
	JobNode *node = (JobNode *)malloc(sizeof(JobNode));

	// if (queue->size >= queue->capacity) {
	// 	double_job_queue_capacity(queue);
	// }

	node->job = job;
    node->next = NULL;

	if (queue->tail == NULL) {
        // queue is empty -> head and tail both become this node
        queue->head = node;
        queue->tail = node;
    } else {
        // add to the end
        queue->tail->next = node;
        queue->tail = node;
    }

    queue->size++;

    // if curr not set yet, start it at the head
    if (queue->curr == NULL) {
        queue->curr = queue->head;
    }
}

/**
 * Remove a job from the queue
 *
 * \param queue  pointer to queue
 * \param job  pointer to the job to be removed
 */
void remove_job_from_queue(JobQueue *queue, Job *job) {
	JobNode *prev = NULL;
    JobNode *currJob = queue->head;

    // find the node containing this job
    while (currJob != NULL && currJob->job != job) {
        prev = currJob;
        currJob = currJob->next;
    }

    // job not found
    if (currJob == NULL) {
        return;
    }

    // unlink curr from the list
    if (prev == NULL) {
        // removing the head
        queue->head = currJob->next;
    } else {
        prev->next = currJob->next;
    }

    if (queue->tail == currJob) {
        // removing the tail
        queue->tail = prev;
    }

    // fix curr: if curr was pointing at this node,
    // move it to the next node, or wrap to the new head
    if (queue->curr == currJob) {
        if (currJob->next != NULL) {
            queue->curr = currJob->next;
        } else {
            queue->curr = queue->head; // if queue empty
        }
    }

    free(currJob);
    queue->size--;

    if (queue->size == 0) {
        // empty queue -> clear everything
        queue->head = queue->tail = queue->curr = NULL;
    }
}

/**
 * A JobQueue iterator (in order of arrival time)
 *
 * \param queue  pointer to queue
 * \return       pointer to the next job, or NULL if the queue is empty
 */
Job *next_job(JobQueue *queue) {
    if (queue->size == 0 || queue->head == NULL) {
        return NULL;
    }

    // if curr empty but queue isn't empty, set to head
    if (queue->curr == NULL) {
        queue->curr = queue->head;
    }

    Job *job = queue->curr->job;

    // advance curr -> move to next node or wrap to head
    if (queue->curr->next != NULL) {
        queue->curr = queue->curr->next;
    } else {
        queue->curr = queue->head;
    }

    return job;
}

/**
 * Free the memory allocated for the job queue
 *
 * \param queue  pointer to queue
 */
void cleanup_job_queue(JobQueue *queue) {
	JobNode *currJob = queue->head;
    while (currJob != NULL) {
        JobNode *next = currJob->next;
        free(currJob);
        currJob = next;
    }

    queue->head = queue->tail = queue->curr = NULL;
    queue->size = 0;
}
