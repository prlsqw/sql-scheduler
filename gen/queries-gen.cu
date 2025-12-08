#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include "../language/headers/grammar.h"
extern "C" {
  #include "../language/headers/executor.h"
  #include "../language/headers/utils.h"
}

__device__ const int NUM_OPS = ARRAY_LEN(QueryOps);
__device__ const int NUM_CMP = ARRAY_LEN(ComparisonOps);

typedef struct {
  int rows;
  int cols;
  Query *output_buff;
  int seed;
  curandState *state;
  int num_digits;
} KernelInput;
// 1D kernel
// 1, query_i
__global__ void generate_queries(KernelInput input) {
  curand_init(input.seed, threadIdx.x, 0, &input.state[threadIdx.x]);
  curandState local_state = input.state[threadIdx.x];
  const int op_i = (int)(NUM_OPS * curand_uniform_double(&local_state));
  const int col_i = (int)(input.cols * curand_uniform_double(&local_state));
  Query output_query = (Query){
    operation : (Operation)op_i,
    column_index : col_i,
    arg1 : -1.0,
    arg2 : -1.0
  };
  if (op_i == Operation::INCREMENT || op_i == Operation::WRITE) {
    output_query.arg1 = pow(10, input.num_digits) * curand_uniform_double(&local_state);
  } else if (op_i == Operation::WRITE_AT || op_i == Operation::COUNT) {
    const int choices = op_i == Operation::WRITE_AT ? input.rows : NUM_CMP;
    output_query.arg1 = (int)(choices * curand_uniform_double(&local_state));
    output_query.arg2 = pow(10, input.num_digits) * curand_uniform_double(&local_state);
  }

  input.output_buff[threadIdx.x] = output_query;
}

int main(int argc, char *argv[]) {
  // get csv path from args
  if (argc < 4 || argc > 5) {
    fprintf(stderr, "Usage: %s <csv_path> <output_path> <num_queries> <seed?>\n", argv[0]);
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

  // get seed
  const int seed = argc == 5 ? atoi(argv[4]) : now();

  // setup input and launch kernel
  KernelInput input = (KernelInput){
      rows: num_rows,
      cols: num_cols,
      output_buff: gpu_output_buff,
      seed: seed,
      state: gpu_state,
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
    // write ops may write numbers that exceed cell length, but parsers/executors will handle this case
    switch (curr.operation) {
      case Operation::INCREMENT:
      case Operation::WRITE:
        fprintf(queries_file_ptr, "%s(%d, %f)\n", QueryOps[curr.operation], curr.column_index, curr.arg1);
        break;
      case Operation::WRITE_AT:
        fprintf(queries_file_ptr, "%s(%d, %d, %f)\n", QueryOps[curr.operation], curr.column_index, (int) curr.arg1, curr.arg2);
        break;
      case Operation::COUNT:
        fprintf(queries_file_ptr, "%s(%d, %s, %f)\n", QueryOps[curr.operation], curr.column_index, ComparisonOps[(int) curr.arg1], curr.arg2);
        break;
      default:
        fprintf(queries_file_ptr, "%s(%d)\n", QueryOps[curr.operation], curr.column_index);
    }
  }
  fclose(queries_file_ptr);
}