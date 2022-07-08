#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --job-name=main_learn_3
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=main_learn_3.out

triton=1  # do not change


# Configuration
simID=3
number_of_samples=400000
operation_mode=1
scenario=1
orthogonality=0
retain=0

module load anaconda
source activate CFmMIMO_PC

python CFmMIMO_PC.py --simulationID $simID --samples $number_of_samples --mode $operation_mode --scenario $scenario --orthogonality $orthogonality --retain $retain --host $triton