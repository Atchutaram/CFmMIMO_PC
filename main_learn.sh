#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem=1M
#SBATCH --job-name=main-learn
#SBATCH --output=script_logs/main_learn.out


first_id=$(sbatch --parsable ./schedulers/setup_sim_params.sh)
second_id=$(sbatch --dependency=afterok:$first_id ./schedulers/schedule_datagen.sh)

# sbatch --dependency=afterok:$second_id ./schedulers/learn.sh
