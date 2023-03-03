import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# TODO:
#   Check file existance 
#   Count n of samples
#   Reuse _main_ code variables and call different training samples sequentially

# temporary hardcoded
cwd = os.getcwd() + "\simID_2\data_logs_training\\betas"
max_index = 8000
half_index = int(max_index / 2) 
matrix_sum1 = 0
matrix_sum2 = 0
betas = []

# read all betas to the list
for f_index in range(max_index):
    f_name = f'betas_sample{f_index}.pt'
    beta_file_path = os.path.join(cwd, f_name)
    m = torch.load(beta_file_path)
    betas.append(np.log(m['betas'].numpy()))
    # betas.append(m['betas'].numpy())

print(betas[0].shape)

# calculate correlation matrix
for i in range(half_index):
    matrix_sum1 += betas[i] @ betas[i].T
    
for i in range(half_index,max_index):
    matrix_sum2 += betas[i] @ betas[i].T
    # matrix_sum += betas[i].T @ betas[i]
    
corr_matrix = matrix_sum1 / half_index
corr_matrix2 = matrix_sum2 / half_index

fig, (ax1, ax2) = plt.subplots(1,2)
sns.heatmap(corr_matrix, ax=ax1)
sns.heatmap(corr_matrix2, ax=ax2)
plt.show()