#include <stdio.h>
#include <stdlib.h>

#define FIRST_COL_NAME "0"
// TODO: make not hard-coded
#define DIGITS 6

__global__ void generate_dataset(char* csv[]) {
  printf("hi, %d %d\n", blockIdx.x, threadIdx.x);
}

// TODO: make extern
int main(int argc, char* argv[]) {
  // get col, row count from args
  if (argc != 4) {
    fprintf(stderr, "Usage: %s <output_path> <num_cols> <num_rows>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  char* output_path = argv[1];
  int num_cols = atoi(argv[2]);
  int num_rows = atoi(argv[3]);

  // construct column name header
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

  // create csv array on gpu
  char** gpu_csv;
  if (cudaMalloc(&gpu_csv, sizeof(char*) * num_rows) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate CSV array on GPU\n");
    exit(EXIT_FAILURE);
  }

  // allocate memory for each line on gpu
  char* gpu_csv_lines[num_rows];
  const int MAX_LINE_LEN = (DIGITS + 1) * num_cols;
  for (int i = 0; i < num_rows; i++) {
    if (cudaMalloc(&gpu_csv_lines[i], sizeof(char) * MAX_LINE_LEN) != cudaSuccess) {
      fprintf(stderr, "Failed to allocate line %d on GPU\n", i);
      exit(EXIT_FAILURE);
    }
  }

  // copy allocated gpu pointers to gpu csv array
  if (cudaMemcpy(gpu_csv, gpu_csv_lines, sizeof(char*) * num_rows, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy line pointers to GPU's CSV array\n");
    exit(EXIT_FAILURE);
  }

  // run kernel on gpu csv
  generate_dataset<<<num_cols, num_rows>>>(gpu_csv);

  // Wait for the kernel to finish
  if(cudaDeviceSynchronize() != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
  }

  // copy back csv array
  char* csv[num_rows + 1];
  csv[0] = col_header;

  // write to file
}