#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <fcntl.h>
#include <unistd.h>

#define FIRST_COL_NAME "0"
#define DEFAULT_SEED 67
#define THREAD_I blockDim.x * blockIdx.x + threadIdx.x
// TODO: make not hard-coded
#define DIGITS 6

// row, column
__global__ void generate_dataset(int* data, int seed, curandState* state) {
  curand_init(seed, THREAD_I, 0, &state[THREAD_I]);
  curandState localState = state[THREAD_I];
  double rand = curand_uniform_double(&localState);
  // curand_uniform_double EXCLUDES 0.0, INCLUDES 1.0, so subtract 1
  int num = ((int) (pow(10.0, DIGITS) * rand)) - 1;
  // data will be row-major
  data[THREAD_I] = num;
}

// TODO: make extern so we can call in tests.c
int main(int argc, char* argv[]) {
  // get col, row count from args
  if (argc < 4 || argc > 5) {
    fprintf(stderr, "Usage: %s <output_path> <num_cols> <num_rows> <seed?>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  char* output_path = argv[1];
  const char* cols_arg = argv[3];
  int num_rows = atoi(argv[2]);
  int num_cols = atoi(cols_arg);
  int seed = argc == 5 ? atoi(argv[4]) : DEFAULT_SEED;

  int num_data = num_rows * num_cols;

  // create data array on gpu
  int* gpu_data;
  if (cudaMalloc(&gpu_data, sizeof(int) * num_data) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate CSV array on GPU\n");
    exit(EXIT_FAILURE);
  }

  // create curand states
  curandState* devStates;
  // TODO: cuRAND state malloc breaks on excessive data, find workaround
  if (cudaMalloc(&devStates, sizeof(curandState) * num_data) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate cuRAND state on GPU\n");
    exit(EXIT_FAILURE);
  }

  printf("generating on gpu...\n");
  // run kernel on gpu csv
  generate_dataset<<<num_rows, num_cols>>>(gpu_data, seed, devStates);

  // start writing csv file
  FILE* csv_file_ptr = fopen(output_path, "w");

  // construct column header
  // first row is 0, ..., n - 1
  int max_name_len = strlen(cols_arg) + 1 + 1;

  fprintf(csv_file_ptr, "%s", FIRST_COL_NAME);
  for (int i = 1; i < num_cols; i++) {
    fprintf(csv_file_ptr, ",%d", i);
  }
  fprintf(csv_file_ptr, "\n");

  // wait for the kernel to finish
  if(cudaDeviceSynchronize() != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
    exit(EXIT_FAILURE);
  }

  // copy back csv array
  int* data = (int*) malloc(sizeof(int) * num_data);
  if (cudaMemcpy(data, gpu_data, sizeof(int) * num_data, cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "Failed to copy data back from GPU\n");
    exit(EXIT_FAILURE);
  }

  printf("writing to file...\n");

  const int print_threshold = num_rows / 10;
  for (int row = 0; row < num_rows; row++) {
    if (row % print_threshold == 0) {
      printf(".");
      fflush(stdout);
    }
    // TODO: leftpad data with zeroes
    fprintf(csv_file_ptr, "%d", data[row * num_cols]);
    for (int col = 1; col < num_cols; col++) {
      fprintf(csv_file_ptr, ",%d", data[row * num_cols + col]);
    }
    fprintf(csv_file_ptr, "\n");
  }
  printf("\ndone writing!\n");

  // cleanup
  free(data);
  fclose(csv_file_ptr);
  cudaFree(gpu_data);
  cudaFree(devStates);
}