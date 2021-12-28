#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem-per-node=1M
#SBATCH --job-name=schedule-datagen
#SBATCH --output=script_logs/schedule_datagen_%a.out
#SBATCH --array=0-4

sim_filename='data_logs_training/params/sim_params_1.pkl'  # do not change this; it is same as given in set_sim_params.py
sample_id=$SLURM_ARRAY_TASK_ID

srun python ./schedule_datagen.py sim_filename sample_id