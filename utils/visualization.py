import matplotlib.pyplot as plt
import os
import torch
import seaborn as sns
import pickle


# Predefined color map for algorithms
algoColorMap = {
    'EPA': 'tab:blue',
    'APG': 'tab:orange',
    'TNN': 'tab:green',
    'FCN': 'tab:red',
    'TDN': 'tab:purple'
}

# Predefined line styles
lineStyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5)), (0, (1, 10))]


def fetchSeValues(resultsFolder, algoList, seMin):
    seOut = [[] for _ in algoList]
    for file in os.listdir(resultsFolder):
        for algoId, algo in enumerate(algoList):
            if algo in file:
                filePathAndName = os.path.join(resultsFolder, file)
                tempArray = torch.load(filePathAndName)
                if seMin:
                    seOut[algoId].append(tempArray['resultSample'].min().unsqueeze(0))
                else:
                    seOut[algoId].append(tempArray['resultSample'])
    
    return seOut

def consolidatedPlots(figIdx, resultsFolders, algoLists, tags, tagsForNonML, plotFolder, seMin):
    # Initialize figures for consolidation
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    
    if tagsForNonML is None:
        tagsForNonML = tags
    elif not (len(tagsForNonML) == len(tags)):
        tempTag = []
        for _ in tags:
            tempTag.append(tagsForNonML[0])
        tagsForNonML = tempTag

    for tagIndex, (resultsFolder, algoList, tag, tagForNonML) in\
            enumerate(zip(resultsFolders, algoLists, tags, tagsForNonML)):
        
        seOut = fetchSeValues(resultsFolder, algoList, seMin)
        
        for algoId, algo in enumerate(algoList):
            color = algoColorMap[algo]
            lineStyle = lineStyles[tagIndex % len(lineStyles)]
            
            seArray = torch.cat(seOut[algoId])
            seOutFinal = seArray.reshape((-1,))
            seSumFinal, _ = seOutFinal.sort()
            
            # Adjust label for clarity
            label = f'{algo} - {tag}'.replace('_', ' ')
            label = f'{algo} - {tag}'.replace('TNN', 'PAPC')
            if algo in ['APG', 'EPA']:
                label = f'{algo} - {tagForNonML}'.replace('_', ' ')

            # Plot CDF with specified color and line style
            ax.plot(
                seSumFinal.cpu().numpy(),
                torch.linspace(0, 1, seSumFinal.size()[0]),
                color=color,
                linestyle=lineStyle,
                label=label
            )

            # Plot PDF with specified color and line style
            sns.kdeplot(
                            seSumFinal.cpu().numpy(),
                            ax=ax2,
                            color=color,
                            linestyle=lineStyle,
                            label=label
                        )
    
    if seMin:
        # Finalizing and saving the consolidated CDF plot
        ax.legend()
        ax.set_xlabel('Worst case user spectral efficiency')
        ax.set_ylabel('CDF')
        cdfPlotFile = os.path.join(plotFolder, f'consolidated_CDF_min_fig{figIdx}.png')
        fig.savefig(cdfPlotFile)

        # Finalizing and saving the consolidated PDF plot
        ax2.legend()
        ax2.set_xlabel('Worst case user spectral efficiency')
        ax2.set_ylabel('PDF')
        pdfPlotFile = os.path.join(plotFolder, f'consolidated_PDF_min_fig{figIdx}.png')
        fig2.savefig(pdfPlotFile)
    else:
        # Finalizing and saving the consolidated CDF plot
        cdfFileName = f'consolidated_CDF_full_fig{figIdx}'
        ax.legend()
        ax.set_xlabel('Per-user spectral efficiency')
        ax.set_ylabel('CDF')
        cdfPlotFile = os.path.join(plotFolder, f'{cdfFileName}.png')
        fig.savefig(cdfPlotFile)
        cdfPlotFilePkl = os.path.join(plotFolder, f'{cdfFileName}.pkl')
        with open(cdfPlotFilePkl, 'wb') as f:
            pickle.dump(fig, f)

        # Finalizing and saving the consolidated PDF plot
        pdfFileName = f'consolidated_PDF_full_fig{figIdx}'
        ax2.legend()
        ax2.set_xlabel('Per-user spectral efficiency')
        ax2.set_ylabel('PDF')
        pdfPlotFile = os.path.join(plotFolder, f'{pdfFileName}.png')
        fig2.savefig(pdfPlotFile)
        pdfPlotFilePkl = os.path.join(plotFolder, f'{pdfFileName}.pkl')
        with open(pdfPlotFilePkl, 'wb') as f:
            pickle.dump(fig2, f)

def localPlotEdits(figIdx, plotFolder, outputFolder):
    
    cdfFileName = f'consolidated_CDF_full_fig{figIdx}'
    cdfPlotFilePkl = os.path.join(plotFolder, f'{cdfFileName}.pkl')
    with open(cdfPlotFilePkl, 'rb') as f:
        fig = pickle.load(f)
    
    pdfFileName = f'consolidated_PDF_full_fig{figIdx}'
    pdfPlotFilePkl = os.path.join(plotFolder, f'{pdfFileName}.pkl')
    with open(pdfPlotFilePkl, 'rb') as f:
        fig2 = pickle.load(f)
    
    # Edit figures here
    
    # Save figures after editing
    cdfPlotFile = os.path.join(outputFolder, f'{cdfFileName}.png')
    fig.savefig(cdfPlotFile)
    
    pdfPlotFile = os.path.join(outputFolder, f'{pdfFileName}.png')
    fig2.savefig(pdfPlotFile)
    
def individualPlots(resultsFolder, algoList, plotFolder, scenario, seMin):

    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()

    scenarioName = f'scenario {scenario}'

    seOut = fetchSeValues(resultsFolder, algoList, seMin)

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
    
    if seMin:
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
    else:
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
    for seMin in [True, False]:
        individualPlots(resultsFolder, algoList, plotFolder, scenario, seMin = seMin)
    plt.show()

def consolidatedPlotter(figIdx, resultsFolders, algoLists, tags, tagsForNonML, plotFolder):
    for seMin in [True, False]:
        consolidatedPlots(
                                figIdx,
                                resultsFolders,
                                algoLists,
                                tags,
                                tagsForNonML,
                                plotFolder,
                                seMin = seMin
                        )
    plt.show()

def localPlotEditor(figIdx, plotFolder, outputFolder):
    localPlotEdits(figIdx, plotFolder, outputFolder)
    plt.show()