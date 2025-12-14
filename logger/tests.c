#include "logger.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
	printf("Initializing logger...\n");
	const char *filename = "local.test_log.csv";
	log_init(filename);

	printf("Logging events...\n");
	log_receive(1, "Job 1 Description");
	sleep(1);

	log_start(1);
	sleep(1);

	log_result(1, "Job 1 Result");
	sleep(1);

	log_stop(1, true);
	sleep(1);

	log_receive(2, "Job 2 Description");
	sleep(1);

	log_start(2);
	sleep(1);

	log_stop(2, false);
	sleep(1);

	log_dump_csv();
	log_destroy();

	FILE *fp = fopen(filename, "r");
	if (!fp) {
		fprintf(stderr, "Failed to open log file: %s\n", filename);
		return 1;
	}

	char line[1024];
	if (fgets(line, sizeof(line), fp)) {
		printf("Header: %s", line);
	}

	int count = 0;
	while (fgets(line, sizeof(line), fp)) {
		printf("Line %d: %s", ++count, line);
	}

	fclose(fp);
	return 0;
}
