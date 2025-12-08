#ifndef SCHEDULER_H
#define SCHEDULER_H 1

#include "../language/language.h"
#include "headers/job_queue.h"
#include <sys/time.h>

#define DEFAULT_QUANTUM 100

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

// A job is a DataFrame along with its query's execution state
typedef struct {
	Dataframe *df;
	ExecutionState *state;
} Job;

// Round Robin, Weighted Round Robin, First In First Out
typedef enum { RR, WRR, FIFO } SchAlgorithm;

#endif
