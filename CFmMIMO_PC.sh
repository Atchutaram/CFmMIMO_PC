#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --job-name=main_learn
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=main_learn.out

triton=1  # do not change


# Training mode
number_of_samples=2000
operation_mode=1
scenario=1
retain=1

# Testing mode
number_of_samples=200
operation_mode=2
scenario=1
retain=1

# Final
number_of_samples=400000
operation_mode=1
scenario=0
retain=1

module load anaconda
source activate CFmMIMO_PC

python CFmMIMO_PC.py --samples $number_of_samples --mode $operation_mode --scenario $scenario --retain $retain --host $triton