#include "headers/logger.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
	printf("Initializing logger...\n");
	log_init();

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

	log_dump_csv("test_log.csv");
	log_destroy();

	FILE *fp = fopen("test_log.csv", "r");
	if (!fp) {
		perror("Failed to open test_log.csv");
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
