#include "../scheduler.h"

// #define INITIAL_JOB_QUEUE_CAPACITY 16

typedef struct{
	Job * job;
	struct JobNode *next;
} JobNode;

typedef struct {
	int size;
	JobNode *head;
	JobNode *tail;
	JobNode *curr;
} JobQueue;

/**
 * Initialize the job queue
 *
 * \param queue  pointer to queue
 */
void initialize_job_queue(JobQueue *queue);

/**
 * Add a job to the queue
 *
 * \param queue  pointer to queue
 * \param job  pointer to the job to be added
 */
void add_job_to_queue(JobQueue *queue, Job *job);

/**
 * Remove a job from the queue
 *
 * \param queue  pointer to queue
 * \param job  pointer to the job to be removed
 */
void remove_job_from_queue(JobQueue *queue, Job *job);

/**
 * A JobQueue iterator
 *
 * \param queue  pointer to queue
 * \return       pointer to the next job, or NULL if the queue is empty
 */
Job *next_job(JobQueue *queue);

/**
 * Free the memory allocated for the job queue
 *
 * \param queue  pointer to queue
 */
void cleanup_job_queue(JobQueue *queue);
