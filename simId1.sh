#!/bin/bash
#SBATCH --time=9:00:00
#SBATCH --job-name=simId1
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=simId1.out
#SBATCH --tmp=1T

triton=1  # do not change
operationMode=1  # do not change


# Configuration
simId=1
scenario=0
retain=0

numberOfSamples=400000
varK=0
randomPilotsFlag=0

module load anaconda
source activate CFmMIMO_PC

python cellFreeMassMimoPowCtrl.py --simulationId $simId --samples $numberOfSamples --mode $operationMode --scenario $scenario --retain $retain --host $triton --varK $varK --randomPilotsFlag $randomPilotsFlag