import os
import sys
import subprocess
import shutil
from metrics import get_metrics
import statistics

def check_file(path):
    if not os.path.exists(path):
        print(f"Error: File '{path}' not found.")
        sys.exit(1)

def run_benchmark(data_file, queries_file):
    check_file(data_file)
    check_file(queries_file)
    check_file("orchestrator")
    
    # Locate feeder.py relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    feeder_path = os.path.join(script_dir, "feeder.py")
    check_file(feeder_path)

    algos = ["FIFO", "RR", "WRR"]
    results = {}

    print(f"Starting Benchmark...")
    print(f"Data Base: {data_file}")
    print(f"Queries:   {queries_file}")
    print("-" * 50)

    for algo in algos:
        log_file = f"temp/local.log_{algo}.csv"
        if os.path.exists(log_file):
            os.remove(log_file)
        
        if os.path.exists("weights"):
            for weight_file in os.listdir("weights"):
                if weight_file.endswith(".weight"):
                    os.remove(os.path.join("weights", weight_file))

        print(f"Running {algo}...")
        
        cmd = f"cat {queries_file} | python3 {feeder_path} 0.05 | ./orchestrator {data_file} {algo} {log_file}"
        
        try:
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"Error running {algo}: {e}")
            continue

        if not os.path.exists(log_file):
            print(f"Error: {log_file} was not created.")
            results[algo] = None
            continue

        stats = get_metrics(log_file)
        results[algo] = stats

    print("\n" + "=" * 85)
    print(f"{'Metric':<25} | {'FIFO':<15} | {'RR':<15} | {'WRR':<15}")
    print("-" * 85)

    def print_row(label, key, multiplier=1.0, fmt=".2f"):
        row = f"{label:<25} | "
        values = []
        for algo in algos:
            res = results.get(algo)
            if res:
                if key in ['count', 'duration_sec', 'throughput']:
                    val = res[key]
                else:
                    val = statistics.mean(res[key])
                values.append(val)
                row += f"{val * multiplier:<15{fmt}} | "
            else:
                values.append(None)
                row += f"{'N/A':<15} | "
        
        print(row)
        return values

    print_row("Throughput (jobs/s)", "throughput")
    print_row("Avg Turnaround (ms)", "turnaround")
    print_row("Avg Response (ms)", "response")
    print_row("Avg Waiting (ms)", "waiting")
    
    print("=" * 85)

    # winner based on Turnaround Time (common metric)
    best_algo = None
    min_turnaround = float('inf')

    for algo, res in results.items():
        if res:
            avg_turn = statistics.mean(res['turnaround'])
            if avg_turn < min_turnaround:
                min_turnaround = avg_turn
                best_algo = algo

    if best_algo:
        print(f"\n🏆 Best Algorithm (Lowest Avg Turnaround): {best_algo}")
    else:
        print("\nCould not determine winner.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 benchmark.py <data_csv> <queries_txt>")
        sys.exit(1)
    
    run_benchmark(sys.argv[1], sys.argv[2])
