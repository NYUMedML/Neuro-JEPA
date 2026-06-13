#!/usr/bin/env python3
import subprocess
import math

batch_size = 1000
length = 100000
num_jobs = math.ceil((length + 1) / batch_size)

# Create a single job script with array
cmd = '''#!/bin/bash
#SBATCH --partition=cpu_dev,cpu_short,cpu_medium,cpu_long,gpu4_dev,gpu4_short,gpu4_medium,gpu4_long,gpu8_medium,gpu8_long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4:00:00
#SBATCH --job-name=preprocess
#SBATCH --output=./logs/log_%a.out
#SBATCH --array=0-{}

source activate head_ct
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# Calculate start and end indices based on array task ID
BATCH_SIZE={}
LENGTH={}
START_IDX=$((SLURM_ARRAY_TASK_ID * BATCH_SIZE))
END_IDX=$((START_IDX + BATCH_SIZE))
if [ $END_IDX -gt $((LENGTH + 1)) ]; then
    END_IDX=$((LENGTH + 1))
fi

python cpu_caching.py --start_idx $START_IDX --end_idx $END_IDX
'''.format(num_jobs - 1, batch_size, length)

# Write the single array job script
with open("./array_job.sh", 'w') as f:
    f.write(cmd)

# Submit the array job
result = subprocess.run("sbatch ./array_job.sh", shell=True, 
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print(result.stdout)
print(f"Submitted array job with {num_jobs} tasks")