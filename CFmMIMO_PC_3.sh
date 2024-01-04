#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --job-name=main_learn_3
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=main_learn_3.out

triton=1  # do not change
operationMode=1  # do not change


# Configuration
simId=3
scenario=3
retain=0
numberOfSamples=100000



module load anaconda
source activate CFmMIMO_PC

python cellFreeMassMimoPowCtrl.py --simulationId $simId --samples $numberOfSamples --mode $operationMode --scenario $scenario --retain $retain --host $triton