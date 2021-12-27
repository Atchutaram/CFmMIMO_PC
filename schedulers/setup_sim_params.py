import pickle
import sys
import os


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def get_sim_params(filename):
    with open(filename, 'rb') as inp:  # Overwrites any existing file.
        return pickle.load(inp)


def setup_sim(argv, triton = True):
    from sim_params import SimulationParameters, OperatingModes
    
    if triton:
        root = os.path.join('//data.triton.aalto.fi', 'work', 'kochark1', 'CFmMIMO_PC_LS')
    else:
        root = os.getcwd()
    
    simulation_parameters = None
    if argv:
        number_of_samples, operation_mode, scenario = argv
        filename = f'sim_params_{scenario}.pkl'
        for mode in OperatingModes:
            if mode == operation_mode:
                operation_mode = mode
        simulation_parameters = SimulationParameters(root, number_of_samples, operation_mode, scenario)
    else:
        filename = 'sim_params_1.pkl'
        simulation_parameters = SimulationParameters(root)
    
    filename = os.path.join(simulation_parameters.params_folder, filename)
    save_object(simulation_parameters, filename)


if __name__ == '__main__':
    argv = sys.argv[1:]
    setup_sim(argv)  # argv = number_of_samples, operation_mode, scenario
else:
    number_of_samples = 4  # also change in schedule_datagen.sh (and others)
    operation_mode = 1  # training_mode: 1
    scenario = 1
    argv = number_of_samples, operation_mode, scenario
    setup_sim(argv, triton = False)

print('Done!')