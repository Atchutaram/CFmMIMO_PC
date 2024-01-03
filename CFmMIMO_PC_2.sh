#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --job-name=main_learn_2
#SBATCH --mem-per-cpu=80G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=main_learn_2.out
#SBATCH --tmp=1T

triton=1  # do not change
operation_mode=1  # do not change


# Configuration
simID=2
scenario=2
retain=0
number_of_samples=9400000



module load anaconda
source activate CFmMIMO_PC

python CFmMIMO_PC.py --simulationID $simID --samples $number_of_samples --mode $operation_mode --scenario $scenario --retain $retain --host $triton
