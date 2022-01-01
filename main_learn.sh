#!/bin/bash

exec >> script_logs/main_learn.out

#module load anaconda
#source activate CF.....

first_id=$(sbatch --parsable setup_sim_params.sh)
second_id=$(sbatch --dependency=afterok:$first_id schedule_datagen.sh)

# sbatch --dependency=afterok:$second_id learn.sh
