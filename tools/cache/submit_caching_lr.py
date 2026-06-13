import pandas as pd
import subprocess
import os
import math

# Configuration
BATCH_SIZE = 2000

JOBS_LIST = [
    {
        "csv_file": "<csv_file_path>",
        "cache_dir": "<cache_dir_path>",
    },
]

def submit_job(job_config):
    csv_file = job_config["csv_file"]
    cache_dir = job_config["cache_dir"]
    
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}")
        return

    # Read CSV length
    print(f"Reading {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
        length = len(df)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return

    # Calculate array range
    if length == 0:
        print(f"Error: CSV file {csv_file} is empty.")
        return

    num_tasks = math.ceil(length / BATCH_SIZE)
    array_range = f"0-{num_tasks - 1}"
    
    print(f"Submitting job for {csv_file}")
    print(f"Dataset length: {length}, Batch size: {BATCH_SIZE}, Tasks: {num_tasks}, Array: {array_range}")

    # Prepare sbatch command
    # We export environment variables to the job
    env_vars = f"ALL,CSV_FILE={csv_file},CACHE_DIR={cache_dir},LENGTH={length},BATCH_SIZE={BATCH_SIZE}"
    
    cmd = [
        "sbatch",
        f"--array={array_range}",
        f"--export={env_vars}",
        "array_job.sh"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Success: {result.stdout.strip()}")
        else:
            print(f"Failed: {result.stderr.strip()}")
    except FileNotFoundError:
        print("Error: 'sbatch' command not found. Are you on a SLURM cluster?")

if __name__ == "__main__":
    for job in JOBS_LIST:
        submit_job(job)
