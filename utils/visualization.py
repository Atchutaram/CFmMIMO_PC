import matplotlib.pyplot as plt
import os
import torch
import seaborn as sns



def performance_plotter(results_folder, algo_list, plot_folder, scenario, model=None):
    # ToDo: model is None and unused for now. This is planned for future
    plt_fns = [plt1, ]  # only one plot as of now
    for fn in plt_fns:
        fn(results_folder, algo_list, plot_folder, scenario)
    plt.show()

def plt1(results_folder, algo_list, plot_folder, scenario):

    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()

    scenario_name = f'scenario_{scenario}'

    se_out = [[] for _ in algo_list]
    for sample_id, file in enumerate(os.listdir(results_folder)):
        for algo_id, algo in enumerate(algo_list):
            if algo in file:
                file_path_and_name = os.path.join(results_folder, file)
                temp_array = torch.load(file_path_and_name)
                se_out[algo_id].append(temp_array['result_sample'])

    for algo_id, algo in enumerate(algo_list):
        if algo == 'ref_algo_two':
            algo = 'APG'
        algo = algo.upper()
        se_array = torch.cat(se_out[algo_id])
        se_out_final = se_array.reshape((-1,))

        se_sum_final, _ = se_out_final.sort()
        
        label = f'{scenario_name} {algo}'.replace('_', ' ')
        ax.plot(se_sum_final.cpu().numpy(), torch.linspace(0, 1, se_sum_final.size()[0]), label=label)

        label = f'{scenario_name} {algo}'.replace('_', ' ')
        ax2 = sns.kdeplot(se_sum_final.cpu().numpy(), label=label)

    ax.legend()
    ax.set_xlabel('Per-user spectral efficiency')
    ax.set_ylabel('CDF')

    plot_file = os.path.join(plot_folder,f'scenario_{scenario}_CDF.png')
    fig.savefig(plot_file)
    
    
    ax2.legend()
    ax2.set_xlabel('Per-user spectral efficiency')
    ax2.set_ylabel('PDF')
    
    plot_file = os.path.join(plot_folder,f'scenario_{scenario}_PDF.png')
    fig2.savefig(plot_file)