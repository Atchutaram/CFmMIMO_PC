import sys
import time

from setup_sim_params import get_sim_params
from channel_env.generate_beta import data_gen


block_width = 200

def run_block(simulation_parameters, block_id):
    for sample_id in range(block_id*block_width, (block_id+1)*block_width):
        data_gen(simulation_parameters, sample_id)

def local_execution():
    
    # Time stamp recording
    start = time.perf_counter()

    import os
    
    sim_filename = os.path.join(os.getcwd(), 'data_logs_training', 'params', 'sim_params_1.pkl')
    simulation_parameters = get_sim_params(sim_filename)
    for block_id in range(4):
        run_block(simulation_parameters, block_id)
    
    # Compute and display execution time.
    finish = time.perf_counter()
    print(f'Finished in {round(finish - start, 2)} second(s)')

if __name__ == '__main__':
    
    # Time stamp recording
    start = time.perf_counter()

    argv = sys.argv[1:]
    if not argv:
        print("Something went wrong!")
        sys.exit()
    
    sim_filename, block_id = argv
    simulation_parameters = get_sim_params(sim_filename)

    run_block(simulation_parameters, int(block_id))
    
    # Compute and display execution time.
    finish = time.perf_counter()
    print(f'Finished in {round(finish - start, 2)} second(s)')