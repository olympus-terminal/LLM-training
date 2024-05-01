#!/bin/bash

#SBATCH --mem=800GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:A100:2,V100:2
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=4
#SBATCH --distribution=cyclic

module load cuda/11.4
module load openmpi/4.1.1

MPI_ARGS="--mca btl_tcp_if_include 10.0.0.0/8 --mca oob_tcp_if_include 10.0.0.0/8"

torchrun --standalone --nnodes ${SLURM_NNODES} --nproc_per_node ${SLURM_NTASKS_PER_NODE} --rdma --node_rank ${SLURM_NODEID} --master_addr ${SLURM_NODELIST} --master_port 1234 --mpi ${MPI_ARGS} python archive_train_wandb-shuffle-downsample.py ../training_set_after_holdout.zip
