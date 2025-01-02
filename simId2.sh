#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --job-name=simId2
#SBATCH --mem-per-cpu=40G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100:1
#SBATCH --output=simId2.out
#SBATCH --tmp=1T

triton=1  # do not change
operationMode=1  # do not change


# Configuration
simId=2
scenario=0
retain=0

numberOfSamples=12000000
varK=0
randomPilotsFlag=0

module load mamba
source activate CFmMIMO_PC

python cellFreeMassMimoPowCtrl.py --simulationId $simId --samples $numberOfSamples --mode $operationMode --scenario $scenario --retain $retain --host $triton --varK $varK --randomPilotsFlag $randomPilotsFlag