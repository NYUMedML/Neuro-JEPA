#!/bin/bash
#SBATCH --partition=cpu_dev,cpu_short,cpu_medium,cpu_long,gpu4_dev,gpu4_short,gpu4_medium,gpu4_long,gpu8_medium,gpu8_long,a100_dev,a100_short,a100_long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=10-00:00:00
#SBATCH --job-name=preprocess
#SBATCH --output=./logs/log_%a.out
# #SBATCH --array=0-200

source activate brain_mri_new
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# Calculate start and end indices based on array task ID
# BATCH_SIZE must be exported/set by the caller or default to 5000
: "${BATCH_SIZE:=5000}"
: "${LENGTH:=1595197}" # Default or from env

START_IDX=$((SLURM_ARRAY_TASK_ID * BATCH_SIZE))
END_IDX=$((START_IDX + BATCH_SIZE))
if [ $END_IDX -gt $((LENGTH + 1)) ]; then
    END_IDX=$((LENGTH + 1))
fi

python cpu_caching.py --start_idx $START_IDX --end_idx $END_IDX --csv_file "$CSV_FILE" --cache_dir "$CACHE_DIR"

