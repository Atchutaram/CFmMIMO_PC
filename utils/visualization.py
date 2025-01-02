import matplotlib.pyplot as plt
import os
import torch
import seaborn as sns
import pickle
import textwrap


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
    ax = fig.gca()
    if figIdx == 1:
        # Retrieve the legend from the axis
        legend = ax.get_legend()

        # Check if the legend exists and has enough items
        if legend is not None and len(legend.get_texts()) >= 8:
            # Retrieve the 7th and 8th legend items
            legend_texts = legend.get_texts()
            
            # Replace text in the 7th and 8th legend items
            seventh_item = legend_texts[6]
            seventh_item.set_text(seventh_item.get_text().replace('9.4', '12'))
            
            eighth_item = legend_texts[7]
            eighth_item.set_text(eighth_item.get_text().replace('9.4', '12'))

    elif figIdx == 2:
        percentile_value = 0.1
        alignmentPointX = 4
        annotationPointSc1 = (alignmentPointX, 0.4)
        annotationPointSc2 = (alignmentPointX, 0.6)
        annotationPointSc3 = (alignmentPointX, 0.8)
        fontsize = 9
        wrap_width = 25
        arrowWidth = 0.75
        
        ax.axhline(y=percentile_value, color='gray', linestyle='--', alpha=0.7)
        ax.text(
            -0.2,  # x-coordinate of the label
            percentile_value+0.035,  # y-coordinate matching the line
            '10th Percentile',  # The label text
            va='center',  # Vertically center the text at the line
            ha='left',    # Align text to the left of the x position
            fontsize=fontsize,  # Adjust the font size as needed
            color='gray'  # Match the label color with the line
        )
        
        # Sc. 1
        x1 = 2.12
        x2 = 3.856
        yDelta = -0.01
        fcnLag = x2 - x1
        
        ax.plot(x1, percentile_value, 'r+')
        ax.plot(x2, percentile_value, 'go')
        
        ax.annotate(
            '', 
            xy = (x1, percentile_value + yDelta),
            xytext = (x2, percentile_value + yDelta),
            arrowprops = dict(arrowstyle='<->', lw=1.5, color=algoColorMap['FCN'])
        )
        xMean = (x1 + x2)/2
        
        
        ax.annotate(
            '', 
            xy = (xMean, percentile_value + yDelta),
            xytext = annotationPointSc1,
            arrowprops = dict(arrowstyle='->', lw=arrowWidth)
        )
        
        x1 = 2.94
        yDelta = -0.02
        epaLag = x2 - x1
        
        ax.plot(x1, percentile_value, 'bx')
        ax.plot(x2, percentile_value, 'go')
        
        ax.annotate(
            '', 
            xy = (x1, percentile_value + yDelta),
            xytext = (x2, percentile_value + yDelta),
            arrowprops = dict(arrowstyle='<->', lw=1.5, color=algoColorMap['EPA'])
        )
        xMeanEPA = (x1 + x2)/2
        
        ax.annotate(
            '', 
            xy = (xMeanEPA, percentile_value + yDelta),
            xytext = annotationPointSc1,
            arrowprops = dict(arrowstyle='->', lw=arrowWidth)
        )
        
        # Prepare the text with auto-wrap
        text = f'PAPC vs FCN and EPA in Sc. 1. Minimum SE seen by top 90% of the users in FCN lags by {fcnLag:.2f} bits/s/Hz while EPA lags by {epaLag:.2f} bits/s/Hz'

        # Wrap the text
        wrapped_text = '\n'.join(textwrap.fill(line, wrap_width) for line in text.split('\n'))

        # Plot with wrapped text
        ax.text(
            *annotationPointSc1,
            wrapped_text,
            ha='left',
            va='top',
            fontsize = fontsize
        )
        
        # Sc. 2
        x1 = 0.246
        x2 = 2.66
        yDelta = 0.01
        fcnLag = x2 - x1
        
        ax.plot(x1, percentile_value, 'r+')
        ax.plot(x2, percentile_value, 'go')
        
        ax.annotate(
            '', 
            xy = (x1, percentile_value + yDelta),
            xytext = (x2, percentile_value + yDelta),
            arrowprops = dict(arrowstyle='<->', lw=1.5, color=algoColorMap['FCN'], linestyle='--')
        )
        xMean = (x1 + x2)/2
        
        ax.annotate(
            '', 
            xy = (xMean, percentile_value + yDelta),
            xytext = annotationPointSc2,
            arrowprops = dict(arrowstyle='->', lw=arrowWidth, linestyle='--')
        )
        
        x1 = 1.74
        yDelta = -0.02
        epaLag = x2 - x1
        
        ax.plot(x1, percentile_value, 'bx')
        ax.plot(x2, percentile_value, 'go')
        
        ax.annotate(
            '', 
            xy = (x1, percentile_value + yDelta),
            xytext = (x2, percentile_value + yDelta),
            arrowprops = dict(arrowstyle='<->', lw=1.5, color=algoColorMap['EPA'], linestyle='--')
        )
        xMeanEPA = (x1 + x2)/2
        
        ax.annotate(
            '', 
            xy = (xMeanEPA, percentile_value + yDelta),
            xytext = annotationPointSc2,
            arrowprops = dict(arrowstyle='->', lw=arrowWidth, linestyle='--')
        )
        
        # Prepare the text with auto-wrap
        text = f'PAPC vs FCN and EPA in Sc. 2. Minimum SE seen by top 90% of the users in FCN lags by {fcnLag:.2f} bits/s/Hz while EPA lags by {epaLag:.2f} bits/s/Hz'

        # Wrap the text
        wrapped_text = '\n'.join(textwrap.fill(line, wrap_width) for line in text.split('\n'))

        # Plot with wrapped text
        ax.text(
            *annotationPointSc2,
            wrapped_text,
            ha='left',
            va='top',
            fontsize = fontsize
        )
        # Sc. 3
        x1 = 0.828
        x2 = 1.74
        yDelta = 0.02
        epaLag = x2 - x1
        
        ax.plot(x1, percentile_value, 'bx')
        ax.plot(x2, percentile_value, 'go')
        
        ax.annotate(
            '', 
            xy = (x1, percentile_value + yDelta),
            xytext = (x2, percentile_value + yDelta),
            arrowprops = dict(arrowstyle='<->', lw=1.5, color=algoColorMap['EPA'], linestyle='-.')
        )
        xMeanEPA = (x1 + x2)/2
        
        ax.annotate(
            '', 
            xy = (xMeanEPA, percentile_value + yDelta),
            xytext = annotationPointSc3,
            arrowprops = dict(arrowstyle='->', lw=arrowWidth, linestyle='-.')
        )
        
        # Prepare the text with auto-wrap
        text = f'PAPC vs EPA in Sc. 3. Minimum SE seen by top 90% of the users in EPA lags by {epaLag:.2f} bits/s/Hz'

        # Wrap the text
        wrapped_text = '\n'.join(textwrap.fill(line, wrap_width) for line in text.split('\n'))

        # Plot with wrapped text
        ax.text(
            *annotationPointSc3,
            wrapped_text,
            ha='left',
            va='top',
            fontsize = fontsize
        )
        
        # #PAPC vs APG
        # # Sc. 2
        # x1 = 2.66
        # x2 = 2.736
        # papcLagSc2 = x2 - x1
        
        # ax.plot(x2, percentile_value, 'x', color='orange')
        # xMean = (x1 + x2)/2
        
        
        # ax.annotate(
        #     '', 
        #     xy = (xMean, percentile_value),
        #     xytext = annotationPointPapc,
        #     arrowprops = dict(arrowstyle='->', lw=arrowWidth, linestyle='--')
        # )
        
        # # Sc. 3
        # x1 = 1.74
        # x2 = 1.817
        # papcLagSc3 = x2 - x1
        
        # ax.plot(x2, percentile_value, 'x', color='orange')
        # xMean = (x1 + x2)/2
        
        
        # ax.annotate(
        #     '', 
        #     xy = (xMean, percentile_value),
        #     xytext = annotationPointPapc,
        #     arrowprops = dict(arrowstyle='->', lw=arrowWidth, linestyle='-.')
        # )
        # text = f'PAPC vs APG in Sc. 2 and Sc. 3. Minimum SE seen by top 90% of the users in PAPC of Sc. 2 lags by {papcLagSc2:.2f} while PAPC of Sc. 3 lags by {papcLagSc3:.2f}'

        # # Wrap the text
        # wrapped_text = '\n'.join(textwrap.fill(line, wrap_width) for line in text.split('\n'))

        # # Plot with wrapped text
        # ax.text(
        #     *annotationPointPapc,
        #     wrapped_text,
        #     ha='left',
        #     va='top',
        #     fontsize = fontsize
        # )
        
        # # ax.set_xlim(right=6)
        ax.legend(loc='upper left')
        fig.set_size_inches(10, 7)


    # Save figures after editing
    cdfPlotFile = os.path.join(outputFolder, f'{cdfFileName}.png')
    fig.savefig(cdfPlotFile, dpi=600, bbox_inches='tight')
    cdfPlotFile = os.path.join(outputFolder, f'{cdfFileName}.pdf')
    fig.savefig(cdfPlotFile, bbox_inches='tight')

    
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