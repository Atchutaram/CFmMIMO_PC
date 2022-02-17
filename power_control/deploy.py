import torch

from .models.nn_setup import CommonParameters



def deploy(model, test_sample, device):
    import pickle

    sc = pickle.load(open(CommonParameters.sc_path, 'rb'))

    with torch.no_grad():
        test_sample = torch.log(test_sample)

        t_shape = test_sample.shape
        test_sample = test_sample.view((-1, CommonParameters.input_size)).to(device='cpu')
        beta_torch = sc.transform(test_sample)[0]
        beta_torch = torch.tensor(beta_torch, device=device, requires_grad=False, dtype=torch.float32).view(t_shape)
        mus_predicted = model(beta_torch)
        return mus_predicted