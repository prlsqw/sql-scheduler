"""
This file is used to run and test the scheduler.
"""

import csv
import os
import random
import subprocess
import sys
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

ORCHESTRATOR_PATH = "orchestrator"
ALGORITHMS = ["FIFO", "RR", "WRR"]
ALL_ALGORITHMS = "ALL"
OUTPUT_DIR = "run"


def generate_run_name():
    devs = ["Avaash", "Devanshu", "Julian", "Shibam"]

    adjectives = [
        "Null",
        "Dead",
        "Hot",
        "Cold",
        "Core",
        "Lost",
        "Async",
        "Lazy",
        "Strict",
        "Greedy",
        "Broken",
        "Fast",
        "Raw",
        "Cursed",
    ]

    nouns = [
        "Loop",
        "Fork",
        "Thread",
        "Stack",
        "Heap",
        "Segv",
        "Crash",
        "Race",
        "Leak",
        "Ping",
        "Patch",
        "Commit",
        "Merge",
        "404",
        "OOM",
    ]

    dev = random.choice(devs)
    adjective = random.choice(adjectives)
    noun = random.choice(nouns)
    output = f"{dev}s_{adjective}_{noun}"
    return output.lower()


def _check_file_exists(file_path: str):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist.")
        sys.exit(1)


def _run_scheduler(
    run_name: str, data_file: str, queries_file: str, algorithm: str
) -> dict[str, str]:
    algorithms = [algorithm] if algorithm != ALL_ALGORITHMS else ALGORITHMS
    log_files = {}

    for algo in algorithms:
        # ensure log directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        log_file = f"{OUTPUT_DIR}/{run_name}_{algo}.csv"

        # cleanup if exists
        if os.path.exists(log_file):
            os.remove(log_file)

        # "recursively" call feeder
        print(f"Running {algo}...")
        cmd = f"cat {queries_file} | python3 run.py feeder | ./orchestrator {data_file} {algo} {log_file}"

        try:
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"Error running {algo}: {e}")
            continue

        if not os.path.exists(log_file):
            print(f"Error: {log_file} was not created.")
            log_files[algo] = None
            continue

        # save log file for metrics
        log_files[algo] = log_file

    return log_files


def _get_metrics(log_file: str) -> dict[str, float]:
    events = []

    # read entire log file into memory
    with open(log_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(
                {
                    "timestamp": int(row["timestamp"]),
                    "id": int(row["id"]),
                    "type": row["type"],
                    "completed": row["completed"] == "true",
                }
            )

    if not events:
        return {}

    # group events by job id
    jobs = defaultdict(list)
    for event in events:
        jobs[event["id"]].append(event)

    # calculate metrics
    turnaround_times = []  # how long it takes to get the job completed
    waiting_times = []  # how long the job is waiting for the CPU
    response_times = []  # how long it takes to get the job initially started
    execution_times = []  # how long the job is running actively in the CPU

    # calculate metrics for each job
    for job_id, job_events in jobs.items():
        job_events.sort(key=lambda x: x["timestamp"])

        receive_event = next((e for e in job_events if e["type"] == "RECEIVE"), None)
        if not receive_event:
            continue

        receive_time = receive_event["timestamp"]

        # turn around time: how long it takes to get the job completed
        # when job is completed - when job is received
        last_stop_event = next(
            (e for e in reversed(job_events) if e["type"] == "STOP" and e["completed"]), None
        )
        if last_stop_event:
            turnaround_times.append(last_stop_event["timestamp"] - receive_time)

        # response time: how long it takes to get the job initially started
        # when job is first started - when job is received
        first_start_event = next((e for e in job_events if e["type"] == "START"), None)
        if first_start_event:
            response_times.append(first_start_event["timestamp"] - receive_time)

        # execution time: how long the job is running actively in the CPU
        # sum up the time the job is running actively in the CPU
        total_exec_time = 0
        current_start = None
        for e in job_events:
            if e["type"] == "START":
                current_start = e["timestamp"]
            elif e["type"] == "STOP" and current_start is not None:
                total_exec_time += e["timestamp"] - current_start
                current_start = None

        if last_stop_event:
            # waiting time = how long the job is waiting for the CPU
            # turnaround time - execution time
            waiting_times.append((last_stop_event["timestamp"] - receive_time) - total_exec_time)
            execution_times.append(total_exec_time)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    def fastest(lst):
        return min(lst) if lst else 0

    def slowest(lst):
        return max(lst) if lst else 0

    # calculate throughput (completed jobs per second)
    completed_jobs = len(turnaround_times)
    total_time = events[-1]["timestamp"] - events[0]["timestamp"]
    if total_time > 0:
        throughput = completed_jobs / (total_time / 1000.0)
    else:
        throughput = 0

    return {
        # turnaround: from receive to last stop (completed)
        "avg_turnaround_ms": avg(turnaround_times),
        # "fastest_turnaround_ms": fastest(turnaround_times),
        # "slowest_turnaround_ms": slowest(turnaround_times),
        # waiting: from receive to first start
        "avg_waiting_ms": avg(waiting_times),
        # "fastest_waiting_ms": fastest(waiting_times),
        # "slowest_waiting_ms": slowest(waiting_times),
        # response: from receive to first start
        "avg_response_ms": avg(response_times),
        # "fastest_response_ms": fastest(response_times),
        # "slowest_response_ms": slowest(response_times),
        # execution: sum of (stop - start)
        "avg_execution_ms": avg(execution_times),
        # "fastest_execution_ms": fastest(execution_times),
        # "slowest_execution_ms": slowest(execution_times),
        # throughput: jobs per second
        "throughput_jobs_sec": throughput,
    }


def _run_evals(log_files: dict[str, str]):
    algo_stats = {}
    for algo, log_file in log_files.items():
        if log_file is None:
            print(f"Error: {algo} log file was not created.")
            continue

        algo_stats[algo] = _get_metrics(log_file)
    return algo_stats


def _print_results(results: dict[str, dict[str, float]]):
    for algo, stats in results.items():
        print(f"\nResults for {algo}:")
        for metric, value in stats.items():
            print(f"  {metric:25}: {value:>10.2f}")


def _display_charts(run_name: str, results: dict[str, dict[str, float]]):
    print("\nGenerating performance charts...")

    # categories to plot
    metrics_to_plot = [
        ("Turnaround Time", "turnaround_ms"),
        ("Waiting Time", "waiting_ms"),
        ("Response Time", "response_ms"),
        ("Execution Time", "execution_ms"),
    ]

    algos = list(results.keys())
    x = np.arange(len(algos))
    width = 0.6  # wider bars since we're only showing averages

    # Create figure with better styling
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Scheduler Performance Comparison", fontsize=20, fontweight="bold", y=0.98)
    axes = []
    for i in range(6):
        axes.append(plt.subplot(2, 3, i + 1))

    # Set figure background
    fig.patch.set_facecolor("#f8f9fa")

    for i, (title, key) in enumerate(metrics_to_plot):
        ax = axes[i]

        avg_vals = [results[algo][f"avg_{key}"] for algo in algos]
        # min_vals = [results[algo][f"fastest_{key}"] for algo in algos]
        # max_vals = [results[algo][f"slowest_{key}"] for algo in algos]

        # Use gradient colors for visual appeal
        colors = ["#4e79a7", "#f28e2c", "#e15759"]
        bars = ax.bar(
            x,
            avg_vals,
            width,
            color=colors[: len(algos)],
            alpha=0.8,
            edgecolor="white",
            linewidth=2,
        )

        # Add value labels with better formatting
        ax.bar_label(bars, padding=3, fmt="%.2f", fontsize=10, fontweight="bold")

        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(algos, fontsize=11, fontweight="bold")
        ax.set_ylabel("Milliseconds", fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.3, color="gray")
        ax.set_facecolor("#ffffff")

        # Add subtle spine styling
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
            spine.set_linewidth(1.5)

    # Plot Throughput with enhanced styling
    ax_tp = axes[4]
    tp_vals = [results[algo]["throughput_jobs_sec"] for algo in algos]
    colors = ["#4e79a7", "#f28e2c", "#e15759"]
    bars_tp = ax_tp.bar(
        algos, tp_vals, color=colors[: len(algos)], alpha=0.8, edgecolor="white", linewidth=2
    )
    ax_tp.bar_label(bars_tp, padding=3, fmt="%.2f", fontsize=10, fontweight="bold")
    ax_tp.set_title("Throughput", fontsize=14, fontweight="bold", pad=10)
    ax_tp.set_ylabel("Jobs / Second", fontsize=11, fontweight="bold")
    ax_tp.set_xlabel("Algorithm", fontsize=11, fontweight="bold")
    ax_tp.grid(axis="y", linestyle="--", alpha=0.3, color="gray")
    ax_tp.set_facecolor("#ffffff")
    for spine in ax_tp.spines.values():
        spine.set_edgecolor("#cccccc")
        spine.set_linewidth(1.5)

    # Hide the empty 6th subplot
    axes[5].axis("off")

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, f"{run_name}_performance_comparison.png")
    plt.savefig(chart_path)
    print(f"Charts saved to: {chart_path}")
    plt.close()


# 1. run all benchmarking code
def run_benchmark(run_name: str, data_file: str, queries_file: str, algorithm: str):
    if not (algorithm in ALGORITHMS or algorithm == ALL_ALGORITHMS):
        print(
            f"Error: Invalid algorithm: {algorithm}. Valid algorithms are: {ALGORITHMS} OR '{ALL_ALGORITHMS}'"
        )
        sys.exit(1)

    print(f"Checking files...")
    _check_file_exists(data_file)
    _check_file_exists(queries_file)
    _check_file_exists(ORCHESTRATOR_PATH)

    print(f"Running scheduler for {algorithm}...")
    log_files = _run_scheduler(run_name, data_file, queries_file, algorithm)

    print(f"Running evals on results...")
    results = _run_evals(log_files)

    print(f"Displaying results...")
    _print_results(results)
    _display_charts(run_name, results)


# 2. run C orchestrator in a subprocess
def feeder():
    delay = 0.1  # delay of 1/10th second

    for line in sys.stdin:
        sys.stdout.write(line)
        sys.stdout.flush()

        if line.strip() == ":quit":
            break

        time.sleep(delay)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "feeder":
        # recursive call to feeder in _run_scheduler
        feeder()
        sys.exit(0)
    elif len(sys.argv) != 4:
        print("Usage: python3 run.py <data_csv> <queries_txt> <algorithm: all | fifo | rr | wrr>")
        sys.exit(1)

    run_name = generate_run_name()
    data_file = sys.argv[1]
    queries_file = sys.argv[2]
    algorithm = sys.argv[3]

    print(f"Running benchmark for {algorithm} as {run_name}...")
    run_benchmark(run_name, data_file, queries_file, algorithm)
