#ifndef LOGGER_H
#define LOGGER_H

#include <stdbool.h>

// log types
typedef enum {
	// used when a new job is recieved
	LOG_RECEIVE,

	// used to log when a job has started execution
	LOG_START,

	// used to log when a job has paused (or completed) execution
	LOG_STOP,

	// used to log the final result of a job
	LOG_RESULT,
} LogType;

typedef struct LogEntry {
	// timestamp of entry
	long timestamp;

	// unique id of job
	int id;

	// type of log entry
	LogType type;

	// description of log entry
	char *description;

	// completed status of log entry
	bool completed;

	// result of log entry
	char *result;

	// pointer to next log entry
	struct LogEntry *next;
} LogEntry;

void log_init();

void log_receive(int id, const char *description);

void log_start(int id);

void log_stop(int id, bool completed);

void log_result(int id, const char *result);

void log_dump_csv(const char *filename);

void log_destroy();

#endif
