#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --mem=1M
#SBATCH --job-name=setup-sim-params
#SBATCH --output=script_logs/setup_sim_params.out

module load anaconda
source activate CFmMIMO_PC

number_of_samples=4  # also change in schedule_datagen.sh (and others)
operation_mode=1  # training_mode: 1
scenario=1

srun python setup_sim_params.py $number_of_samples $operation_mode $scenario