import matplotlib.pyplot as plt
import os
import torch
import seaborn as sns

def fetchSeValues(resultsFolder, algoList, seMin=False):
    seOut = [[] for _ in algoList]
    for file in os.listdir(resultsFolder):
        for algoId, algo in enumerate(algoList):
            # print(algo)
            if algo in file:
                filePathAndName = os.path.join(resultsFolder, file)
                tempArray = torch.load(filePathAndName)
                if seMin:
                    seOut[algoId].append(tempArray['resultSample'].min().unsqueeze(0))
                    # tt = 1
                    # if algoId == tt:
                    #     print(f'{algo}: {type(tempArray["resultSample"])}')
                else:
                    seOut[algoId].append(tempArray['resultSample'])
    
    return seOut

def minCDFPlots(resultsFolder, algoList, plotFolder, scenario):

    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()

    scenarioName = f'scenario {scenario}'

    seOut = fetchSeValues(resultsFolder, algoList, seMin=True)

    for algoId, algo in enumerate(algoList):
        algo = algo.upper()
        seArray = torch.cat(seOut[algoId])
        seOutFinal = seArray.reshape((-1,))

        seSumFinal, _ = seOutFinal.sort()
        
        label = f'{scenarioName} {algo}'.replace('_', ' ')
        ax.plot(
                    seSumFinal.cpu().numpy(),
                    torch.linspace(0, 1, seSumFinal.size()[0]),
                    label=label
                )

        label = f'{scenarioName} {algo}'.replace('_', ' ')
        ax2 = sns.kdeplot(seSumFinal.cpu().numpy(), label=label)

    ax.legend()
    ax.set_xlabel('Worst case user spectral efficiency')
    ax.set_ylabel('CDF')

    plotFile = os.path.join(plotFolder,f'scenario{scenario}CDF_min.png')
    fig.savefig(plotFile)
    
    
    ax2.legend()
    ax2.set_xlabel('Worst case user spectral efficiency')
    ax2.set_ylabel('PDF')
    
    plotFile = os.path.join(plotFolder,f'scenario{scenario}PDF_min.png')
    fig2.savefig(plotFile)

def fullCDFPlots(resultsFolder, algoList, plotFolder, scenario):

    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()

    scenarioName = f'scenario {scenario}'

    seOut = fetchSeValues(resultsFolder, algoList)

    for algoId, algo in enumerate(algoList):
        algo = algo.upper()
        seArray = torch.cat(seOut[algoId])
        seOutFinal = seArray.reshape((-1,))

        seSumFinal, _ = seOutFinal.sort()
        
        label = f'{scenarioName} {algo}'.replace('_', ' ')
        ax.plot(
                    seSumFinal.cpu().numpy(),
                    torch.linspace(0, 1, seSumFinal.size()[0]),
                    label=label
                )

        label = f'{scenarioName} {algo}'.replace('_', ' ')
        ax2 = sns.kdeplot(seSumFinal.cpu().numpy(), label=label)

    ax.legend()
    ax.set_xlabel('Per-user spectral efficiency')
    ax.set_ylabel('CDF')

    plotFile = os.path.join(plotFolder,f'scenario{scenario}CDF_full.png')
    fig.savefig(plotFile)
    
    
    ax2.legend()
    ax2.set_xlabel('Per-user spectral efficiency')
    ax2.set_ylabel('PDF')
    
    plotFile = os.path.join(plotFolder,f'scenario{scenario}PDF_full.png')
    fig2.savefig(plotFile)
    
    
def performancePlotter(resultsFolder, algoList, plotFolder, scenario):
    pltFns = [fullCDFPlots, minCDFPlots]  # only one plot as of now
    for fn in pltFns:
        fn(resultsFolder, algoList, plotFolder, scenario)
    plt.show()
