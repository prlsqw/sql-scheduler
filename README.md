# SQL Commands Scheduling

## Tasks
1. Set up SQLite in C, define a small set of possible commands our system accepts (Domain Specific Language, DSL).
2. Build ability to generate *large* random datasets and SQL commands from our DSL. 
3. Build a naive Round Robin scheduler to run the commands.
4. Build a custom, weighted-quantum Round Robin scheduler to run commands.
    - Build a perceptron to estimate command execution length (in ms).
      - On initialization, use functions from step 1 to generate sample of SQL queries,
      - Run them individually sequentially without corrupting raw data,
      - Build a dataset from runtime lengths:

        | SQL Command   | Runtime | Data Size  | Type Access |
        |---------------|---------|------------|-------------|
        | READ x FROM y | 30ms    | (n x m)    | int         |
        | WRITE x TO y  | 60ms    | (n x m)    | string      |
        | READ x FROM y | 30ms    | (n x m)    | float       |

      - Perceptron builds weights from past data.
    - Round Robin based on weights from perceptron.
5. Build an evaluator that generates metrics.


## User input might look like the following
```[bash]
COMMAND   |   ARRIVAL TIME
COMMAND   |   ARRIVAL TIME
COMMAND   |   ARRIVAL TIME
COMMAND   |   ARRIVAL TIME
```

## Concepts from class
1. Scheduling
2. Processes/GPU (for the evaluator to build datasets and SQL commands)
3. File I/O (for the evaluator to store things)
4. GPU (to run the perceptron on)
5. [Optionally] Thread Synchronization (systematically prevent race conditions in the scheduler)

## What could go wrong
- SQLite might not suit our needs, might need a different interface for DB access (potentially Python),
- The Custom Round Robin Scheduler performs worse than regular RR,
- DSL definition might be too big to be tractable or too small to be useful,
- Context swapping might be too slow/frequent to be useful.
