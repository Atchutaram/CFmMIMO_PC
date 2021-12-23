import pickle
import sys
import os

default_file_name = os.path.join('//data.triton.aalto.fi', 'work', 'kochark1', 'CFmMIMO_PC_LS', 'data_logs', 'sim_params', 'sim_params_1.pkl')

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def get_sim_params(filename):
    with open(filename, 'rb') as inp:  # Overwrites any existing file.
        return pickle.load(inp)


def setup_sim(argv):
    from sim_params import SimulationParameters, OperatingModes
    if argv:
        number_of_samples, operation_mode, scenario, filename = argv
        for mode in OperatingModes:
            if mode == operation_mode:
                operation_mode = mode
        simulation_parameters = SimulationParameters(number_of_samples, operation_mode, scenario)
    else:
        simulation_parameters = SimulationParameters()
        filename = default_file_name
    save_object(simulation_parameters, filename)


if __name__ == '__main__':
    argv = sys.argv[1:]
    setup_sim(argv)  # argv = number_of_samples, operation_mode, scenario, filename