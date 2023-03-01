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
    # print(m['betas'].numpy())
    # betas.append(m['betas'].to(dtype=torch.float32))
    betas.append(m['betas'].numpy())

# calculate correlation matrix
for i in range(max_index):
    matrix_sum += betas[i] @ betas[i].T
    
print(matrix_sum)
corr_matrix = matrix_sum / max_index
print("-----------------")
print(corr_matrix)

print(len(betas))


# beta_original = m['betas'].to(dtype=torch.float32)
# pilot_sequence = m['pilot_sequence'].to(dtype=torch.int32)





# print(beta_original)


# x = np.array(beta_original)
# print(x.shape)

# x = np.loadtxt(fname)


# files = next(os.walk(path))

# file_path_and_name = os.path.join(simulation_parameters.data_folder, f'betas_sample{sample_id}.pt')


#     def __init__(self, data_path, phi_orth, normalizer, mode, n_samples):
#         self.path = data_path
#         _, _, files = next(os.walk(self.path))
#         self.n_samples = min(len(list(filter(lambda k: 'betas' in k, files))), n_samples)
#         self.sc = normalizer
#         self.mode = mode
#         self.phi_orth = phi_orth
        
#     def __getitem__(self, index):
#         beta_file_name = f'betas_sample{index}.pt'
#         beta_file_path = os.path.join(self.path, beta_file_name)
#         m = torch.load(beta_file_path)
#         beta_original = m['betas'].to(dtype=torch.float32)
#         pilot_sequence = m['pilot_sequence'].to(dtype=torch.int32)

#         phi = torch.index_select(self.phi_orth, 0, pilot_sequence)
#         phi_cross_mat = torch.abs(phi.conj() @ phi.T)
        
#         beta_torch = torch.log(beta_original)
#         beta_torch = beta_torch.to(dtype=torch.float32)
#         beta_torch = beta_torch.reshape(beta_original.shape)

#         return phi_cross_mat, beta_torch, beta_original




# print(x)