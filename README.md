# SQL Scheduler

A job scheduling system for executing parallel SQL-like queries on CSV datasets with multiple scheduling algorithms. The project implements a complete query language processor with support for concurrent query execution, scheduling, and performance metrics.

Have a new idea for a new scheduling algorithm? Implement it here: [scheduler/scheduler.c](scheduler/scheduler.c).

## Building and Running

### Prerequisites

- `clang` or compatible C compiler
- `nvcc` (NVIDIA CUDA compiler) for data generation
- `python3` with `matplotlib` and `numpy` for benchmarking
- CUDA-capable GPU (for data generation only)

### Build Commands

**Build everything (recommended):**

```bash
make
```

**Build and run full benchmark (is slow):**

```bash
make run
```

This generates datasets, queries, and runs benchmarks for all algorithms.

**Call the REPL:**

```bash
make demo
```

Launches the REPL with sample data.

### Configuration (in Makefile)

- `INITIAL_SEED`: Random seed for reproducible data generation
- `DATASET_SIZE`: Dataset dimensions `rows x cols x digits`
- `QUERIES_SIZE`: Number of queries to generate
- `DEFAULT_QUANTUM_MS`: Time slice for scheduling

## Overview

This system provides a custom query language for data analytics operations on CSV files. It features:

- **Custom SQL-like DSL** for data operations
- **Multiple scheduling algorithms** (Round Robin, Weighted Round Robin, FIFO)
- **Weighted scheduling** that learns from query execution patterns
- **Logging** to CSVs for performance analysis
- **CUDA-accelerated data generation** for testing and benchmarks
- **REPL** for manual query execution
- **Automated benchmarking** with performance visualization

## Project Structure

### Core Libraries

#### `language/`

The query language parser and executor. This library defines a custom SQL-like DSL and implements:

- **Grammar** ([headers/grammar.h](language/headers/grammar.h)): Defines the query language operations and syntax
- **Parser** ([parser.c](language/parser.c)): Parses query strings into structured Query objects
- **Executor** ([executor.c](language/executor.c)): Executes queries on CSV dataframes with resumable execution states

**Syntax:**

```
AVERAGE(column)
MEDIAN(column)
INCREMENT(column, value)
WRITE(column, value)
WRITE_AT(column, row, value)
COUNT(column, op, value)     # op = < | l | ! | = | g | >
```

The interpreter is preemption aware. Queries can be paused mid-execution and resumed later and the interpreter makes sure to maintain correct state across preemptions.

#### `scheduler/`

Job scheduler with multiple algorithms:

- **Scheduler** ([scheduler.c](scheduler/scheduler.c)): Core scheduling logic implementing RR, WRR, and FIFO algorithms
- **Job Queue** ([job_queue.c](scheduler/job_queue.c)): Thread-safe circular queue for managing pending jobs
- **Secretary** ([secretary.c](scheduler/secretary.c)): Scheduler initialization and job submission interface
- **Weights** ([weights.c](scheduler/weights.c)): weight calculation based on historical execution times

**Scheduling Algorithms:**

- **FIFO (First In First Out)**: Jobs execute to completion in arrival order
- **RR (Round Robin)**: Each job gets equal time slice (default: 100ms quantum)
- **WRR (Weighted Round Robin)**: Time slices adjusted based on historical execution time of each operation type (currently, it scales the default quantum of 100ms by some factor derived from past execution times)

The scheduler runs in a dedicated pthread, pulling jobs from the queue and executing them according to the selected algorithm.

#### `logger/`

Thread-safe event logging system:

- **Logger** ([logger.c](logger/logger.c)): Maintains a linked list of timestamped events
- Records job lifecycle: RECEIVE, START, STOP (with completion status), RESULT
- Exports logs to CSV for analysis with timestamps, job IDs, event types, and completion status
- Used by benchmarking system to calculate turnaround time, waiting time, response time, and throughput

#### `gen/`

Data generators for creating test datasets and queries using CUDA:

- **Dataset Generator** ([dataset-gen.cu](gen/dataset-gen.cu)): Generates large CSV files with random data using curand
- **Query Generator** ([queries-gen.cu](gen/queries-gen.cu)): Creates randomized queries

## Entry Points

### `orchestrator` - REPL

[orchestrator.c](orchestrator.c) provides an interactive command-line interface for executing queries:

```bash
./orchestrator <data_file.csv> <RR|WRR|FIFO> <output_log.csv>
```

**Example:**

```bash
./orchestrator data/set-10000x20x10x42.csv WRR logs/run-1.csv
>>> AVERAGE(5)
AVERAGE(5): 123.456
>>> COUNT(3, >, 500)
COUNT(3, >, 500): 42
>>> :quit
```

### `run.py` - Automated Benchmarking

[run.py](run.py) orchestrates automated performance testing across all scheduling algorithms.

## Performance Metrics

The benchmarking system calculates:

- **Turnaround Time**: Total time from job submission to completion
- **Waiting Time**: Time job spends waiting in queue
- **Response Time**: Time from submission to first execution
- **Execution Time**: Actual CPU time spent processing
- **Throughput**: Jobs completed per second

Results are visualized in comparison charts showing relative performance of each scheduling algorithm.
