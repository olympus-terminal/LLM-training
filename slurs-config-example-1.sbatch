#!/bin/bash

#SBATCH --mem=800GB
#SBATCH --time=96:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20

### these are good resource requests for training a 370m model, batch size=96 and max_seq_length=2048
