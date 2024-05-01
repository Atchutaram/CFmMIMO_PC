import torch
from .modes import OperatingModes


class SystemParameters:
    
    def __init__(self, simulationParameters):
        defaultModels = ['TNN', 'FCN', ]
        if simulationParameters.scenario==0:
            coverageArea = 0.01  # in sq.Km
            maxNumberOfUsers = 4
            accessPointDensity = 1000
            
            models = defaultModels

        elif simulationParameters.scenario==1:
            coverageArea = 0.1  # in sq.Km
            maxNumberOfUsers = 20
            accessPointDensity = 1000

            models = defaultModels
            
        elif simulationParameters.scenario==2:
            coverageArea = 0.1
            maxNumberOfUsers = 40
            accessPointDensity = 1000

            models = defaultModels
            
        elif simulationParameters.scenario==3:
            defaultModels = ['TNN',]
            coverageArea = 0.1
            maxNumberOfUsers = 80
            accessPointDensity = 1000

            models = defaultModels
            
        # elif simulationParameters.scenario==4:
        #     coverageArea = 1
        #     maxNumberOfUsers = 500
        #     accessPointDensity = 2000
        #     models = ['TNN',]  # Plan is to do ['AE-FCN', 'TNN']
        else:
            raise('Invalid Scenario Configuration')


        self.param_L = torch.tensor(
                                        140.715087,
                                        device=simulationParameters.device,
                                        requires_grad=False,
                                        dtype=torch.float32
                                    )
        self.d0 = torch.tensor(
                                    0.01,
                                    device=simulationParameters.device,
                                    requires_grad=False,
                                    dtype=torch.float32
                            )
        self.d1 = torch.tensor(
                                    0.05,
                                    device=simulationParameters.device,
                                    requires_grad=False,
                                    dtype=torch.float32
                            )
        self.sigma_sh = 8  # in dB
        self.bandWidth = 20e6  # in Hz
        self.noiseFigure = 9  # in dB
        self.zeta_d = 1  # in W
        self.zeta_p = 0.2  # in W
        self.log_d0 = torch.log10(self.d0)
        self.log_d1 = torch.log10(self.d1)
        
        self.tau = 3
        

        self.No_Hz = -173.975
        self.totalNoisePower = 10 ** ((self.No_Hz - 30) / 10) * self.bandWidth\
            * 10 ** (self.noiseFigure / 10)
        self.zeta_d /= self.totalNoisePower
        self.zeta_p /= self.totalNoisePower

        self.Tp = 20
        self.Tc = 200

        
        self.numberOfAntennas = 4  # N
        self.coverageArea = torch.tensor(
                                            coverageArea,
                                            requires_grad=False,
                                            device=simulationParameters.device,
                                            dtype=torch.float32
                                        )  # D
        self.maxNumberOfUsers = maxNumberOfUsers  # K
        
        self.minNumberOfUsers = self.maxNumberOfUsers
        if simulationParameters.varyingNumberOfUsersFlag:
            self.minNumberOfUsers = (self.maxNumberOfUsers//2)
            
        self.accessPointDensity = accessPointDensity
        self.models = models
        simulationParameters.handleModelSubFolders(self.models)

        self.areaWidth = torch.sqrt(self.coverageArea)  # in Km
        self.areaHeight = self.areaWidth
        self.numberOfAccessPoints = round(accessPointDensity*self.coverageArea.item())  # M
        print(f"""Number of APs: {self.numberOfAccessPoints}
Number of users: {self.maxNumberOfUsers}""")

        torch.manual_seed(seed=0)
        randomMat = torch.normal(0, 1, (self.Tp, self.Tp))
        self.phiOrth, _, _ = torch.linalg.svd(randomMat)

        areaDims = torch.tensor(
                                    [self.areaWidth, self.areaHeight],
                                    device=simulationParameters.device,
                                    requires_grad=False,
                                    dtype=torch.float32
                                )
        
        torch.manual_seed(seed=2)
        randVec = torch.rand(
                                (2, self.numberOfAccessPoints),
                                device=simulationParameters.device,
                                requires_grad=False, dtype=torch.float32
                            ) - 0.5  # 2 X M
        
        apPositions = torch.einsum('d,dm->md ', areaDims, randVec).to(simulationParameters.device)
        
        aw = self.areaWidth
        ah = self.areaHeight
        refList = [
                        [  0,   0],
                        [-aw,   0],
                        [  0, -ah],
                        [ aw,   0],
                        [  0,  ah],
                        [-aw,  ah],
                        [ aw, -ah],
                        [-aw, -ah],
                        [ aw,  ah]
                ]
        ref = torch.tensor(
                                refList,
                                device=simulationParameters.device,
                                requires_grad=False,
                                dtype=torch.float32
                        )
        
        self.apMinusRef = apPositions.view(-1, 1, 2)-ref