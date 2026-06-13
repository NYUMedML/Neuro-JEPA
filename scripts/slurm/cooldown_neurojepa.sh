#!/bin/bash
#SBATCH --job-name=neurojepa-base-cooldown
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=15-00:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=gl40s_long
#SBATCH --output=./log/neurojepa-base-cooldown.out
#SBATCH --error=./log/neurojepa-base-cooldown.err

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$((10000 + ($SLURM_JOB_ID % 50000)))
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=^lo,docker0,virbr0
export PYTHONUNBUFFERED=1

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

# Activate your environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate neurojepa_env

#module load cuda/11.8
module load gcc/10.2.0

export PYTHONPATH=/path/to/Neuro-JEPA/src

# Run the srun command which executes the script on all allocated nodes
srun --cpu_bind=v --accel-bind=gn bash -c '
  # Each node will run this block
  export WORLD_SIZE=$SLURM_NTASKS
  export RANK=$SLURM_PROCID

  echo "STARTING ON NODE $SLURMD_NODENAME: RANK $RANK of $WORLD_SIZE"
  
  python -m torch.distributed.run \
    --nproc_per_node $SLURM_GPUS_ON_NODE \
    --nnodes $SLURM_NNODES \
    --node_rank $SLURM_NODEID \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    ../pretrain.py \
    -cd /path/to/Neuro-JEPA/configs/pretrain \
    --config-name cooldown_neurojepa_base
'