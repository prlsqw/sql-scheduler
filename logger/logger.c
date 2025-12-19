#include "logger.h"
#include "../language/language.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static LogEntry *dummy_head = NULL;
static LogEntry *curr = NULL;
static char *filename = NULL;
bool is_logger_initialized = false;
static pthread_mutex_t logger_lock = PTHREAD_MUTEX_INITIALIZER;

/**
 * Initializes logger for the first time.
 * 
 * \param log_filename	Filename for output log
 */
void log_init(const char *log_filename) {
	if (is_logger_initialized) {
		perror("Logger already initialized");
		exit(EXIT_FAILURE);
	}

	// check if output log file can be opened for writing
	FILE *log_file = fopen(log_filename, "w");
	if (log_file == NULL) {
		fprintf(stderr, "Cannot open log file: %s\n", log_filename);
		exit(EXIT_FAILURE);
	}
	fclose(log_file);

	dummy_head = (LogEntry *)malloc(sizeof(LogEntry));
	if (!dummy_head) {
		perror("Failed to allocate memory for logger initialization");
		exit(EXIT_FAILURE);
	}

	dummy_head->next = NULL;
	curr = dummy_head;
	filename = strdup(log_filename);
	is_logger_initialized = true;
}

/**
 * Creates new log entry
 * 
 * \param id		ID for job
 * \param type	Log type for entry
 * \return	Heap-allocated LogEntry object.
 */
static LogEntry *create_entry(int id, LogType type) {
	// space for new entry
	LogEntry *entry = (LogEntry *)malloc(sizeof(LogEntry));
	if (!entry) {
		perror("Failed to allocate memory for log entry");
		exit(EXIT_FAILURE);
	}

	// given values
	entry->timestamp = (long)now(); // current time
	entry->id = id;					// job id
	entry->type = type;				// type of log

	// default values
	entry->description = NULL; // description of log
	entry->completed = false;  // completed status
	entry->result = NULL;	   // result of log
	entry->next = NULL;		   // next entry

	return entry;
}

/**
 * Adds entry to the logger linked list
 * 
 * \param entry	Populated, heap-allocated LogEntry object
 */
static void append_entry(LogEntry *entry) {
	// since this function is used by every other logging function, this check
	// can have a single instance here for all logging functions
	if (!is_logger_initialized) {
		perror("Logger not initialized");
		exit(EXIT_FAILURE);
	}

	pthread_mutex_lock(&logger_lock);
	curr->next = entry;
	curr = entry;
	pthread_mutex_unlock(&logger_lock);
}

/**
 * Logs a new job as just received.
 * 
 * \param id					ID of job
 * \param	description	Description of entry
 */
void log_receive(int id, const char *description) {
	LogEntry *entry = create_entry(id, LOG_RECEIVE);
	if (description) {
		entry->description = strdup(description);
	}
	append_entry(entry);
}

/**
 * Logs when job starts execution
 * 
 * \param id					ID of job
 */
void log_start(int id) {
	LogEntry *entry = create_entry(id, LOG_START);
	append_entry(entry);
}

/**
 * Logs when job stops execution
 * 
 * \param id				ID of job
 * \param	completed	If job is completed or not
 */
void log_stop(int id, bool completed) {
	LogEntry *entry = create_entry(id, LOG_STOP);
	entry->completed = completed;
	append_entry(entry);
}

/**
 * Logs final result of job
 * 
 * \param id			ID of job
 * \param	result	Result of job
 */
void log_result(int id, const char *result) {
	LogEntry *entry = create_entry(id, LOG_RESULT);
	if (result) {
		entry->result = strdup(result);
	}
	append_entry(entry);
}

/**
 * Dumps current logs into CSV file.
 */
void log_dump_csv() {
	if (!is_logger_initialized) {
		perror("Logger not initialized");
		exit(EXIT_FAILURE);
	}

	FILE *file = fopen(filename, "w");
	if (!file) {
		fprintf(stderr, "Failed to open log file (%s) for writing\n", filename);
		exit(EXIT_FAILURE);
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

/**
 * Cleans up all logs.
 */
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

	// reset dummy_head, curr, filename, and is_logger_initialized
	dummy_head = NULL;
	curr = NULL;
	filename = NULL;
	is_logger_initialized = false;
}
