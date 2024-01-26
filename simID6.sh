#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --job-name=simId6
#SBATCH --mem-per-cpu=80G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=simId6.out
#SBATCH --tmp=1T

triton=1  # do not change
operationMode=1  # do not change


# Configuration
simId=6
scenario=3
retain=0

numberOfSamples=1000000
varK=1
randomPilotsFlag=0

module load anaconda
source activate CFmMIMO_PC

python cellFreeMassMimoPowCtrl.py --simulationId $simId --samples $numberOfSamples --mode $operationMode --scenario $scenario --retain $retain --host $triton --varK varK --randomPilotsFlag randomPilotsFlag
