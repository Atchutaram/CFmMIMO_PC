from pytorch_lightning import Trainer, loggers as plLoggers
import torch

from .models.utils import initializeHyperParams, loadTheLatestModelAndParamsIfExists
from .gradientHandler import grads


def train(simulationParameters, systemParameters):
    modelsList = systemParameters.models

    for modelName in modelsList:
        initializeHyperParams(modelName, simulationParameters, systemParameters)
        model = loadTheLatestModelAndParamsIfExists(modelName, None, systemParameters, grads)

        print(simulationParameters.resultsBase, modelName)
        tbLogger = plLoggers.TensorBoardLogger(
                                                    save_dir=simulationParameters.resultsBase,
                                                    name=modelName
                                                )
        if torch.cuda.is_available()>0:
            trainer = Trainer(
                                accelerator="gpu",
                                devices=-1, max_epochs=model.numEpochs,
                                logger=tbLogger
                            )
        else:
            trainer = Trainer(max_epochs=model.numEpochs, logger=tbLogger)
        
        trainer.fit(model)
