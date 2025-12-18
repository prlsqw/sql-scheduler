#include "scheduler.h"
#include <sys/stat.h>

double get_operation_quantum(Dataframe *df, Query *query,
							 double baseline_quantum) {
	double total_time = 0.0;
	double time_tally = 0;
	int count_tally = 0;

	// fseek the file to start reading from the beginning
	fseek(df->weights, 0, SEEK_SET);

	// read all weights to calculate total time
	int N = ARRAY_LEN(QueryOps);
	for (int i = 0; i < N; i++) {
		double op_time;
		int op_count;

		fscanf(df->weights, "%lf %d", &op_time, &op_count);
		total_time += op_time;

		if (i == query->operation) {
			time_tally = op_time;
			count_tally = op_count;
		}
	}

	// weight is a number between 0 and 1 representing the relative time
	// taken by this operation compared to others
	double this_op_avg = count_tally ? time_tally / count_tally : 0;
	double all_ops_avg = total_time / N;

	if (all_ops_avg == 0) {
		return baseline_quantum;
	}

	double weight = (this_op_avg + all_ops_avg) / all_ops_avg;

	// quantum time is scaled linearly based on this weight
	return weight * baseline_quantum;
}

void update_operation_weight(Dataframe *df, Query *query,
							 double observed_time) {
	// skip the first query->operation entries to reach the one we want
	fseek(df->weights,
		  // + 2 for: ' ' and '\n' in each line
		  query->operation * (WEIGHT_TIME_WIDTH + WEIGHT_COUNT_WIDTH + 2),
		  SEEK_SET);

	// read current time and count
	double current_time;
	int current_count;
	fscanf(df->weights, "%lf %d", &current_time, &current_count);

	// update time and count
	current_time += observed_time;
	current_count += 1;

	// seek back to the start of this operation's entry
	fseek(df->weights, -1 * (WEIGHT_TIME_WIDTH + WEIGHT_COUNT_WIDTH + 2),
		  SEEK_CUR);

	fprintf(df->weights, "%-*lf %-*d\n", WEIGHT_TIME_WIDTH, current_time,
			WEIGHT_COUNT_WIDTH, current_count);
}
