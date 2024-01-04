#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --job-name=main_learn_0
#SBATCH --mem-per-cpu=80G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=main_learn_0.out
#SBATCH --tmp=1T

triton=1  # do not change
operationMode=1  # do not change


# Configuration
simId=0
scenario=0
retain=0
numberOfSamples=9400000

module load anaconda
source activate CFmMIMO_PC

python cellFreeMassMimoPowCtrl.py --simulationId $simId --samples $numberOfSamples --mode $operationMode --scenario $scenario --retain $retain --host $triton
