#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem=1M
#SBATCH --job-name=learn
#SBATCH --output=./../script_logs/learn.out

module load anaconda
source activate CFmMIMO_PC

sim_filename = './../data_logs/sim_params/sim_params_1.pkl'  # do not change this; it is same as given in set_sim_params.py

srun python learn.py sim_filename model_folder