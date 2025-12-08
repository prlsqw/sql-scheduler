#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <fcntl.h>
#include <unistd.h>
extern "C" {
  #include "../language/headers/utils.h"
}

#define DEFAULT_SEED (now())
#define THREAD_I blockDim.x * blockIdx.x + threadIdx.x

// row, column
__global__ void generate_dataset(int* data, int seed, int digits, curandState* state) {
  curand_init(seed, THREAD_I, 0, &state[THREAD_I]);
  curandState localState = state[THREAD_I];
  double rand = curand_uniform_double(&localState);
  // curand_uniform_double EXCLUDES 0.0, INCLUDES 1.0, so subtract 1
  int num = ((int) (pow(10.0, digits) * rand)) - 1;
  // data will be row-major
  data[THREAD_I] = num;
}

// TODO: make extern so we can call in tests.c
int main(int argc, char* argv[]) {
  // get col, row count from args
  if (argc < 5 || argc > 6) {
    fprintf(stderr, "Usage: %s <output_path> <num_rows> <num_cols> <digits> <seed?>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  char* output_path = argv[1];
  int num_rows = atoi(argv[2]);
  const char* cols_arg = argv[3];
  int num_cols = atoi(cols_arg);
  int digits = atoi(argv[4]);
  int seed = argc == 6 ? atoi(argv[5]) : DEFAULT_SEED;

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

  // run kernel on gpu csv
  generate_dataset<<<num_rows, num_cols>>>(gpu_data, seed, digits, devStates);

  // start writing csv file
  FILE* csv_file_ptr = fopen(output_path, "w");

  // construct column header
  // first row is 0, ..., n - 1
  int max_name_len = strlen(cols_arg) + 1 + 1;

  for (int i = 0; i < num_cols - 1; i++) {
    fprintf(csv_file_ptr, "%d,", i);
  }
  fprintf(csv_file_ptr, "%d\n", num_cols - 1);

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

  for (int row = 0; row < num_rows; row++) {
    // TODO: leftpad data with zeroes
    fprintf(csv_file_ptr, "%d", data[row * num_cols]);
    for (int col = 1; col < num_cols; col++) {
      fprintf(csv_file_ptr, ",%d", data[row * num_cols + col]);
    }
    fprintf(csv_file_ptr, "\n");
  }

  // cleanup
  free(data);
  fclose(csv_file_ptr);
  cudaFree(gpu_data);
  cudaFree(devStates);
}