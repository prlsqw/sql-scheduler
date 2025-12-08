#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include "../language/headers/grammar.h"
extern "C" {
  #include "../language/headers/executor.h"
}

#define ARRAY_LEN(array) sizeof(array) / sizeof(array[0])

typedef struct {
  int rows;
  int cols;
  Query *output_buff;
  int seed;
  curandState *state;
  int num_ops;
  int num_cmp;
  int num_digits;
} KernelInput;
// 1D kernel
// 1, query_i
__global__ void generate_queries(KernelInput input) {
  curand_init(input.seed, threadIdx.x, 0, &input.state[threadIdx.x]);
  curandState local_state = input.state[threadIdx.x];
  const int op_i = (int)(input.num_ops * curand_uniform_double(&local_state));
  const int col_i = (int)(input.cols * curand_uniform_double(&local_state));
  Query output_query = (Query){
    operation : (Operation)op_i,
    column_index : col_i,
    arg1 : -1.0,
    arg2 : -1.0
  };
  if (op_i >= Operation::INCREMENT && op_i <= Operation::WRITE) {
    output_query.arg1 = pow(10, input.num_digits) * curand_uniform_double(&local_state);
  } else if (op_i >= Operation::WRITE_AT) {
    const int choices = op_i == Operation::WRITE_AT ? input.rows : input.num_cmp;
    output_query.arg1 = (int)(choices * curand_uniform_double(&local_state));
    output_query.arg2 = pow(10, input.num_digits) * curand_uniform_double(&local_state);
  }

  input.output_buff[threadIdx.x] = output_query;
}

// returns maximum length of an Operation
int max_operation_len() {
  int max = strlen(OpArray[0]);
  int curr;
  for (int i = 1; i < ARRAY_LEN(OpArray); i++) {
    curr = strlen(OpArray[i]);
    if (curr > max)
      max = curr;
  }
  return max;
}

int main(int argc, char *argv[]) {
  // get csv path from args
  if (argc != 4) {
    fprintf(stderr, "Usage: %s <csv_path> <output_path> <num_queries>\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  const char *csv_path = argv[1];
  Dataframe df;
  initialize(&df, csv_path);
  // save data from dataframe
  const int num_rows = df.num_rows;
  const int num_cols = df.num_cols;
  const int num_digits = df.cell_length;

  cleanup(&df);

  const char *num_queries_arg = argv[3];
  const int NUM_QUERIES = atoi(num_queries_arg);
  // setup output buffer
  Query *gpu_output_buff;
  const int OUTPUT_BUFF_SIZE = sizeof(Query) * NUM_QUERIES;
  if (cudaMalloc(&gpu_output_buff, OUTPUT_BUFF_SIZE) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate output buffer on GPU\n");
    exit(EXIT_FAILURE);
  }

  // setup curand states
  curandState *gpu_state;
  if (cudaMalloc(&gpu_state, sizeof(curandState) * NUM_QUERIES) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate cuRAND state on GPU\n");
    exit(EXIT_FAILURE);
  }

  KernelInput input = (KernelInput){
      rows: num_rows,
      cols: num_cols,
      output_buff: gpu_output_buff,
      seed: 67,
      state: gpu_state,
      num_ops: ARRAY_LEN(OpArray),
      num_cmp: ARRAY_LEN(ComparisonOps),
      num_digits: num_digits
    };
  generate_queries<<<1, NUM_QUERIES>>>(input);

  // get file pointer to output file
  const char *output_path = argv[2];
  FILE *queries_file_ptr = fopen(output_path, "w");
  if (queries_file_ptr == NULL) {
    perror("Error writing to output file");
    exit(EXIT_FAILURE);
  }

  // wait for the kernel to finish
  if (cudaDeviceSynchronize() != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
    exit(EXIT_FAILURE);
  }

  // copy back query array from gpu
  Query *queries = (Query *)malloc(OUTPUT_BUFF_SIZE);
  if (cudaMemcpy(queries, gpu_output_buff, OUTPUT_BUFF_SIZE, cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "Failed to copy data back from GPU\n");
    exit(EXIT_FAILURE);
  }

  // start writing to file
  Query curr;
  for (int i = 0; i < NUM_QUERIES; i++) {
    curr = queries[i];
    // TODO: write ops may write numbers that exceed cell length
    switch (curr.operation) {
      case Operation::INCREMENT:
      case Operation::WRITE:
        fprintf(queries_file_ptr, "%s(%d, %f)\n", OpArray[curr.operation], curr.column_index, curr.arg1, curr.arg2);
        break;
      case Operation::WRITE_AT:
        fprintf(queries_file_ptr, "%s(%d, %d, %f)\n", OpArray[curr.operation], curr.column_index, (int) curr.arg1, curr.arg2);
        break;
      case Operation::COUNT:
        fprintf(queries_file_ptr, "%s(%d, %s, %f)\n", OpArray[curr.operation], curr.column_index, ComparisonOps[(int) curr.arg1], curr.arg2);
        break;
      default:
        fprintf(queries_file_ptr, "%s(%d)\n", OpArray[curr.operation], curr.column_index, curr.arg1, curr.arg2);
    }
  }
  fclose(queries_file_ptr);
}