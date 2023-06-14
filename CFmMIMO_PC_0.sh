#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=main_learn_0
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=main_learn_0.out

triton=1  # do not change
operation_mode=1  # do not change


# Configuration
simID=0
scenario=0
retain=0
number_of_samples=400000

module load anaconda
source activate CFmMIMO_PC

python CFmMIMO_PC.py --simulationID $simID --samples $number_of_samples --mode $operation_mode --scenario $scenario --retain $retain --host $triton