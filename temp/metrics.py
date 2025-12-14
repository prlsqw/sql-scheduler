import csv
import sys
import statistics

def get_metrics(log_file):
    jobs = {}
    
    try:
        with open(log_file, 'r') as f:
            reader = csv.DictReader(f)
            
            first_timestamp = None
            last_timestamp = None

            for row in reader:
                ts = int(row['timestamp'])
                job_id = row['id']
                event_type = row['type']
                
                if first_timestamp is None:
                    first_timestamp = ts
                last_timestamp = ts

                if job_id not in jobs:
                    jobs[job_id] = {
                        'receive': None,
                        'first_start': None,
                        'completion': None,
                        'exec_intervals': [],
                        'current_start': None
                    }
                
                job = jobs[job_id]

                if event_type == 'RECEIVE':
                    job['receive'] = ts
                
                elif event_type == 'START':
                    if job['first_start'] is None:
                        job['first_start'] = ts
                    job['current_start'] = ts
                
                elif event_type == 'STOP':
                    if job['current_start'] is not None:
                        job['exec_intervals'].append((job['current_start'], ts))
                        job['current_start'] = None
                    
                    if row['completed'] == 'true':
                        job['completion'] = ts

    except FileNotFoundError:
        print(f"Error: File '{log_file}' not found.")
        return None

    turnaround_times = []
    response_times = []
    waiting_times = []
    total_execution_times = []

    completed_jobs_count = 0

    for job_id, job in jobs.items():
        if job['completion'] is None:
            continue
            
        completed_jobs_count += 1
        
        turnaround = job['completion'] - job['receive']
        turnaround_times.append(turnaround)
        
        response = job['first_start'] - job['receive']
        response_times.append(response)
        
        total_exec = sum(stop - start for start, stop in job['exec_intervals'])
        total_execution_times.append(total_exec)
        
        waiting = turnaround - total_exec
        waiting_times.append(waiting)

    if completed_jobs_count == 0:
        return None

    duration_ms = last_timestamp - first_timestamp
    duration_sec = duration_ms / 1000.0
    throughput = completed_jobs_count / duration_sec if duration_sec > 0 else 0
    
    return {
        "count": completed_jobs_count,
        "duration_sec": duration_sec,
        "throughput": throughput,
        "turnaround": turnaround_times,
        "response": response_times,
        "waiting": waiting_times,
        "execution": total_execution_times
    }

def print_metrics(log_file, metrics):
    if not metrics:
        print(f"No completed jobs found in {log_file}.")
        return

    print(f"--- Scheduling Metrics for {log_file} ---")
    print(f"Total Jobs Completed: {metrics['count']}")
    print(f"Total Simulation Time: {metrics['duration_sec']:.2f} seconds")
    print(f"Throughput: {metrics['throughput']:.2f} jobs/second")
    print("-" * 30)
    
    print(f"{'Metric':<20} | {'Average (ms)':<15} | {'Min (ms)':<10} | {'Max (ms)':<10}")
    print("-" * 65)
    
    data_points = [
        ("Turnaround Time", metrics['turnaround']),
        ("Response Time", metrics['response']),
        ("Waiting Time", metrics['waiting']),
        ("Execution Time", metrics['execution'])
    ]

    for name, data in data_points:
        avg = statistics.mean(data)
        min_val = min(data)
        max_val = max(data)
        print(f"{name:<20} | {avg:<15.2f} | {min_val:<10} | {max_val:<10}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 metrics.py <log_file.csv>")
        sys.exit(1)
    
    m = get_metrics(sys.argv[1])
    print_metrics(sys.argv[1], m)
