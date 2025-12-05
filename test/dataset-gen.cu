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
// TODO: doesn't work with excessive amt of rows or cols
int main(int argc, char* argv[]) {
  // get col, row count from args
  if (argc < 4 || argc > 5) {
    fprintf(stderr, "Usage: %s <output_path> <num_cols> <num_rows> <seed?>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  char* output_path = argv[1];
  int num_rows = atoi(argv[2]);
  int num_cols = atoi(argv[3]);
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
  if (cudaMalloc(&devStates, sizeof(curandState) * num_data) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate cuRAND state on GPU\n");
    exit(EXIT_FAILURE);
  }

  // run kernel on gpu csv
  generate_dataset<<<num_rows, num_cols>>>(gpu_data, seed, devStates);


  // start writing csv file
  int csv_fd = open(output_path, O_CREAT|O_WRONLY|O_TRUNC, S_IRUSR|S_IWUSR);

  // construct column header
  // first row is 0, ..., n
  int max_name_len = strlen(argv[2]) + 1;
  int col_header_len = max_name_len * num_cols;
  char* col_header = (char*) malloc(sizeof(char) * col_header_len);

  strcpy(col_header, FIRST_COL_NAME);
  char next_col_name[max_name_len + 1];
  for (int i = 1; i < num_cols; i++) {
    sprintf(next_col_name, ",%d", i);
    strcat(col_header, next_col_name);
  }
  strcat(col_header, "\n");

  const int MAX_LINE_LEN = (DIGITS + 1) * num_cols + 2;
  char file_buffer[col_header_len + MAX_LINE_LEN * num_rows + 100];
  strcat(file_buffer, col_header);

  // wait for the kernel to finish
  if(cudaDeviceSynchronize() != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
    exit(EXIT_FAILURE);
  }

  // copy back csv array
  int data[num_rows][num_cols];
  if (cudaMemcpy(data, gpu_data, sizeof(int) * num_data, cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "Failed to copy data back from GPU\n");
    exit(EXIT_FAILURE);
  }
  char buffer[MAX_LINE_LEN];
  char temp[DIGITS + 1];

  for (int row = 0; row < num_rows; row++) {
    sprintf(buffer, "%d", data[row][0]);
    for (int col = 1; col < num_cols; col++) {
      sprintf(temp, ",%d", data[row][col]);
      strcat(buffer, temp);
    }
    strcat(buffer, "\n");
    
    strcat(file_buffer, buffer);
  }

  write(csv_fd, file_buffer, strlen(file_buffer));

  // cleanup
  close(csv_fd);

  free(col_header);
  cudaFree(gpu_data);
  cudaFree(devStates);
}