import os
import torch
import numpy as np


# TODO:
#   Check file existance 
#   Count n of samples

# temporary hardcoded
cwd = os.getcwd() + "\simID_2\data_logs_training\\betas"
max_index = 10
matrix_sum = 0
betas = []

# read all betas to list
for f_index in range(max_index):
    f_name = f'betas_sample{f_index}.pt'
    beta_file_path = os.path.join(cwd, f_name)
    m = torch.load(beta_file_path)
    betas.append(m['betas'].numpy())

# calculate correlation matrix
for i in range(max_index):
    matrix_sum += betas[i] @ betas[i].T
    
corr_matrix = matrix_sum / max_index

