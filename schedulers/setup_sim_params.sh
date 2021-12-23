#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --mem=1M
#SBATCH --job-name=setup-sim-params
#SBATCH --output=./../script_logs/setup_sim_params.out


number_of_samples=4
operation_mode=1  # training_mode: 1

scenario=1
filename='./../data_logs/sim_params/sim_params_1.pkl'  # do not change this; it is same as given in set_sim_params.py

srun python setup_sim_params.py number_of_samples operation_mode scenario filename