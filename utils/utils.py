import pickle
import glob
import os
import time
import shutil
import sys


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as inp:  # Overwrites any existing file.
        return pickle.load(inp)


def delete_folder(*args):
    for folder in args:
        for _ in range(5):
            if not os.path.exists(folder):
                break
            shutil.rmtree(folder, ignore_errors=False, onerror=None)
            time.sleep(0.5)

        if os.path.exists(folder):
            print(f"\n'{folder}' folder was not deleted")
            sys.exit()


def handle_deletion_and_creation(folder, number_of_samples=None, retain=False, force_retain= False):
    if os.path.exists(folder):
        if force_retain:
            return
        old_number_of_samples = len(os.listdir(folder))
        if retain:
            if old_number_of_samples==number_of_samples:
                return
            else:
                response = query_fn(f"""
Retain Failed!
You request to retain {number_of_samples} number of samples while we have {old_number_of_samples} samples in the folder {folder}.
Do you want to overwrite the data folder [y/n]? """)
                if response == 'n':
                    print(f'Data folder retain cannot be performed! Either set the set the --samples option to {old_number_of_samples} or --retain option to 0.')
                    sys.exit()

    delete_folder(folder)
    
    os.mkdir(folder)

def delete_folder_contents(grad_inps_folder):

    files = glob.glob(os.path.join(grad_inps_folder, '*'))
    for f in files:
        for index in range(1000):
            try:
                os.remove(f)
            except:
                pass
            if not os.path.exists(f):
                break
            if index == 999:
                raise Exception(f"Sorry, could not delete {f}")


def query_fn(message):
    while True:
        query = input(message)
        response = query[0].lower()
        if query == '' or response not in ['y', 'n']:
            print('Please answer with [y/n]!')
        else:
            break
    return response

def find_the_latest_file(model_folder):
    import glob
    file = None
    list_of_files = glob.glob(os.path.join(model_folder, '*'))
    if list_of_files:
        file = max(list_of_files, key=os.path.getctime)
    if file is not None:
        if not os.path.isfile(file):
            file = None
    return file

def tensor_max_min_print(tensor, text, exit_flag=False):
    print(f'Min of {text}: {tensor.min().detach().item()} Max of {text}: {tensor.max().detach().item()} \n')
    if exit_flag:
        sys.exit()