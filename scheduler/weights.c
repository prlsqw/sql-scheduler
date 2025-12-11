#include "scheduler.h"
#include <sys/stat.h>

double get_operation_quantum(Dataframe *df, Query *query) {
	double total_time = 0.0;
	double time_tally = 0;
	int count_tally = 0;

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

	double weight = (this_op_avg + all_ops_avg) / all_ops_avg;

	// quantum time is scaled linearly based on this weight
	return weight * DEFAULT_QUANTUM_MS;
}
