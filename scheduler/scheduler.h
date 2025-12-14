#ifndef SCHEDULER_H
#define SCHEDULER_H 1

#include "../language/language.h"
#include <sys/time.h>
#include <pthread.h>

#define DEFAULT_QUANTUM_MS 100

// A job is a DataFrame along with its query's execution state
typedef struct {
	Dataframe *df;
	ExecutionState *state;
} Job;

// include at the bottom so it can see Job definition
#include "headers/job_queue.h"

// Scheduler def needs to know about JobQueue
typedef struct {
	// queue of jobs to be scheduled
	JobQueue queue;

	// dataframe this scheduler is managing
	Dataframe *df;

	time_t quantum_ms;

	// max life; scheduler stops after this time if no new queries arrive
	time_t max_life_ms;

	// scheduling algorithm
	void (*algorithm)(JobQueue *queue, time_t quantum, time_t max_life_ms);

	// Scheduler thread handler
	pthread_t thread;
} Scheduler;

// Round Robin, Weighted Round Robin, First In First Out
typedef enum { RR, WRR, FIFO } SchAlgorithm;

// Secretary functions
void initialize_scheduler(Scheduler *scheduler, time_t quantum,
						  SchAlgorithm algorithm, Dataframe *df);

int schedule_query(Scheduler *scheduler, Query *query);

void cleanup_scheduler(Scheduler *scheduler);

// Weights functions
double get_operation_quantum(Dataframe *df, Query *query,
							 double baseline_quantum);

void update_operation_weight(Dataframe *df, Query *query, double observed_time);

// Scheduler algorithms
void rr_scheduler(JobQueue *queue, time_t quantum, time_t max_life_ms);

void wrr_scheduler(JobQueue *queue, time_t quantum, time_t max_life_ms);

void fifo_scheduler(JobQueue *queue, time_t quantum, time_t max_life_ms);

/**
 * Worker Function for Scheduling Scheduler
 * \param arg      Pointer to scheduler
 */
void *scheduler_thread_main(void *arg);

#endif
