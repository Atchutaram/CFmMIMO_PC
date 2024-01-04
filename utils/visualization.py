import matplotlib.pyplot as plt
import os
import torch
import seaborn as sns


def fullCDFPlots(resultsFolder, algoList, plotFolder, scenario):

    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()

    scenarioName = f'scenario{scenario}'

    seOut = [[] for _ in algoList]
    for file in os.listdir(resultsFolder):
        for algoId, algo in enumerate(algoList):
            if algo in file:
                filePathAndName = os.path.join(resultsFolder, file)
                tempArray = torch.load(filePathAndName)
                seOut[algoId].append(tempArray['resultSample'])

    for algoId, algo in enumerate(algoList):
        if algo == 'refAlgoTwo':
            algo = 'APG'
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

    plotFile = os.path.join(plotFolder,f'scenario{scenario}CDF.png')
    fig.savefig(plotFile)
    
    
    ax2.legend()
    ax2.set_xlabel('Per-user spectral efficiency')
    ax2.set_ylabel('PDF')
    
    plotFile = os.path.join(plotFolder,f'scenario{scenario}PDF.png')
    fig2.savefig(plotFile)
    
    
def performancePlotter(resultsFolder, algoList, plotFolder, scenario):
    pltFns = [fullCDFPlots, ]  # only one plot as of now
    for fn in pltFns:
        fn(resultsFolder, algoList, plotFolder, scenario)
    plt.show()
