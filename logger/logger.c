#include "logger.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static LogEntry *dummy_head = NULL;
static LogEntry *curr = NULL;

// initialize logger for the first time
void log_init() {
	dummy_head = (LogEntry *)malloc(sizeof(LogEntry));
	if (!dummy_head) {
		perror("Failed to allocate memory for logger initialization");
		exit(EXIT_FAILURE);
	}

	dummy_head->next = NULL;
	curr = dummy_head;
}

// create new log entry
static LogEntry *create_entry(int id, LogType type) {
	// space for new entry
	LogEntry *entry = (LogEntry *)malloc(sizeof(LogEntry));
	if (!entry) {
		perror("Failed to allocate memory for log entry");
		exit(EXIT_FAILURE);
	}

	// given values
	entry->timestamp = (long)time(NULL); // current time
	entry->id = id;						 // job id
	entry->type = type;					 // type of log

	// default values
	entry->description = NULL; // description of log
	entry->completed = false;  // completed status
	entry->result = NULL;	   // result of log
	entry->next = NULL;		   // next entry

	return entry;
}

// add entry to the logger linked list
static void append_entry(LogEntry *entry) {
	curr->next = entry;
	curr = entry;
}

// log a new job (just received)
void log_receive(int id, const char *description) {
	LogEntry *entry = create_entry(id, LOG_RECEIVE);
	if (description) {
		entry->description = strdup(description);
	}
	append_entry(entry);
}

// log when job starts execution
void log_start(int id) {
	LogEntry *entry = create_entry(id, LOG_START);
	append_entry(entry);
}

// log when job stops execution
void log_stop(int id, bool completed) {
	LogEntry *entry = create_entry(id, LOG_STOP);
	entry->completed = completed;
	append_entry(entry);
}

// log final result of job
void log_result(int id, const char *result) {
	LogEntry *entry = create_entry(id, LOG_RESULT);
	if (result) {
		entry->result = strdup(result);
	}
	append_entry(entry);
}

void log_dump_csv(const char *filename) {
	FILE *file = fopen(filename, "w");
	if (!file) {
		perror("Failed to open log file for writing");
		return;
	}

	// write csv header
	fprintf(file, "timestamp,id,type,description,completed,result\n");

	LogEntry *current = dummy_head->next; // start from the first entry

	// write each log entry as row in csv
	while (current) {
		// 1. entry timestamp
		long timestamp = current->timestamp;
		fprintf(file, "%ld", timestamp);

		// 2. entry job id
		int id = current->id;
		fprintf(file, ",%d", id);

		// 3. entry type
		const char *type_str = "UNKNOWN";
		switch (current->type) {
			case LOG_RECEIVE:
				type_str = "RECEIVE";
				break;
			case LOG_START:
				type_str = "START";
				break;
			case LOG_STOP:
				type_str = "STOP";
				break;
			case LOG_RESULT:
				type_str = "RESULT";
				break;
		}
		fprintf(file, ",%s", type_str);

		// 4. entry description
		const char *description = current->description;
		if (description) {
			// cover with quotes to handle commas
			fprintf(file, ",\"%s\"", description);
		} else {
			// no description
			fprintf(file, ",");
		}

		// 5. entry completed status
		bool completed = current->completed;
		if (strcmp(type_str, "STOP") == 0) {
			// only store completed status for stop entries
			fprintf(file, ",%s", completed ? "true" : "false");
		} else {
			// no completed outcome
			fprintf(file, ",");
		}

		// 6. entry result
		const char *result = current->result;
		if (result) {
			// cover with quotes to handle commas
			fprintf(file, ",\"%s\"", result);
		} else {
			// no result
			fprintf(file, ",");
		}

		// 7. new line
		fprintf(file, "\n");
		current = current->next;
	}

	// close opened file
	fclose(file);
}

void log_destroy() {
	LogEntry *current = dummy_head;

	// loop through all entries and free memory
	while (current) {
		LogEntry *temp = current;
		current = current->next;
		if (temp->description) {
			free(temp->description);
		}
		if (temp->result) {
			free(temp->result);
		}
		free(temp);
	}

	// reset dummy_head and curr pointer
	dummy_head = NULL;
	curr = NULL;
}
