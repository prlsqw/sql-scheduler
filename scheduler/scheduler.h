#ifndef SCHEDULER_H
#define SCHEDULER_H 1

#include "../language/language.h"
#include <sys/time.h>

#define DEFAULT_QUANTUM 100

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

	// in ms
	time_t quantum;

	// max life; scheduler stops after this time if no new queries arrive
	time_t max_life;

	// scheduling algorithm
	void (*algorithm)(JobQueue *queue, time_t quantum, time_t max_life);
} Scheduler;

// Round Robin, Weighted Round Robin, First In First Out
typedef enum { RR, WRR, FIFO } SchAlgorithm;

#endif
