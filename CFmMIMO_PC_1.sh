#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --job-name=main_learn
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --output=main_learn.out

triton=1  # do not change


# Configuration
simID=1
number_of_samples=400
operation_mode=1
scenario=0
orthogonality=1
retain=0

module load anaconda
source activate CFmMIMO_PC

python CFmMIMO_PC.py --simulationID $simID --samples $number_of_samples --mode $operation_mode --scenario $scenario --orthogonality $orthogonality --retain $retain --host $triton