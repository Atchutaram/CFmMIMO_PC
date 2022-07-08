#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --job-name=main_learn_1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=main_learn_1.out

triton=1  # do not change


# Configuration
simID=1
number_of_samples=400000
operation_mode=1
scenario=0
orthogonality=1
retain=0

module load anaconda
source activate CFmMIMO_PC

python CFmMIMO_PC.py --simulationID $simID --samples $number_of_samples --mode $operation_mode --scenario $scenario --orthogonality $orthogonality --retain $retain --host $triton