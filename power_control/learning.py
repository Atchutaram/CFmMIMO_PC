from .setup import CommonParameters
from .utils import load_the_latest_model_and_params_if_exists


def train(device, model_folder):
    CommonParameters.data_preprocessing()
    model = load_the_latest_model_and_params_if_exists(model_folder, device)
    num_epochs = CommonParameters.num_epochs
    train_loader = model.train_loader()
    
    if CommonParameters.VARYING_STEP_SIZE:
        opt, _ = model.configure_optimizers()
    else:
        opt = model.configure_optimizers()

    for epoch_id in range(num_epochs):
        for beta_torch, beta_original in train_loader:
            utility = model.training_step(beta_torch, beta_original, opt, epoch_id)
        print(epoch_id, -utility.mean())