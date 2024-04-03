import pickle
import glob
import os
import time
import shutil
import sys
from sys import exit
import platform
import psutil
import cpuinfo  # Import the cpuinfo library


def logSystemInfoAndLatency(simulationParameters, avgLatency):
    osInfo = platform.platform()
    platformCpuInfo = platform.processor()
    cpuinfoCpuInfo = cpuinfo.get_cpu_info().get('brand_raw', 'Unknown Processor')
    ramInfo = psutil.virtual_memory().total / (1024 ** 3)

    fileName = "systemInfo"
    if simulationParameters.varyingNumberOfUsersFlag:
        fileName = fileName + "_varK"
    if simulationParameters.minNumberOfUsersFlag:
        fileName = fileName + "_minK"
    
    fileName = fileName + ".txt"

    systemInfo = f"Operating System: {osInfo}\n" \
                 f"Processor (Platform): {platformCpuInfo}\n" \
                 f"Processor (CPUInfo): {cpuinfoCpuInfo}\n" \
                 f"RAM: {ramInfo:.2f} GB\n"\
                 f"avgLatency: {avgLatency}\n"

    filePath = os.path.join(simulationParameters.resultsBase, fileName)
    with open(filePath, "w") as file:
        file.write(systemInfo)


def cleanFolders():
    """ Clean old simulation folders and exit"""
    import glob
    from utils.utils import deleteFolder
    
    dirs = glob.glob("simId*/")
    deleteFolder(*dirs)
    
    dirs = glob.glob("consolidatedResults/")
    deleteFolder(*dirs)
    
    print(f"Cleaned all! ")
    exit()

def saveObject(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def loadObject(filename):
    with open(filename, 'rb') as inp:  # Overwrites any existing file.
        return pickle.load(inp)


def deleteFolder(*args):
    for folder in args:
        for _ in range(5):
            if not os.path.exists(folder):
                break
            shutil.rmtree(folder, ignore_errors=False, onerror=None)
            time.sleep(0.5)

        if os.path.exists(folder):
            print(f"\n'{folder}' folder was not deleted")
            sys.exit()


def handleDeletionAndCreation(folder, numberOfSamples=None, retain=False, forceRetain= False):
    if os.path.exists(folder):
        if forceRetain:
            return
        oldNumberOfSamples = len(os.listdir(folder))
        if retain:
            if oldNumberOfSamples==numberOfSamples:
                return
            else:
                response = queryFn(f"""
Retain Failed!
You request is to retain {numberOfSamples} number of samples,
while we have {oldNumberOfSamples} samples in the folder {folder}.
Do you want to overwrite the data folder [y/n]? """)
                if response == 'n':
                    print(f('Retaining operation for the data folder cannot be performed!'
                            'Either set the set the --samples option to {oldNumberOfSamples}'
                            'or --retain option to 0.'))
                    sys.exit()
    
    import random
    random.seed()
    time.sleep(random.uniform(1, 20))
    if os.path.exists(folder) and forceRetain:
        return
    
    deleteFolder(folder)
    
    os.mkdir(folder)

def deleteFolderContents(gradInpsFolder):

    files = glob.glob(os.path.join(gradInpsFolder, '*'))
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


def queryFn(message):
    while True:
        query = input(message)
        response = query[0].lower()
        if query == '' or response not in ['y', 'n']:
            print('Please answer with [y/n]!')
        else:
            break
    return response

def findTheLatestFile(modelFolder):
    import glob
    file = None
    listOfFiles = glob.glob(os.path.join(modelFolder, '*'))
    if listOfFiles:
        file = max(listOfFiles, key=os.path.getctime)
    if file is not None:
        if not os.path.isfile(file):
            file = None
    return file

def findTheLatestFolder(parentFolder):
    import glob
    latestFolder = max(glob.glob(os.path.join(parentFolder, '*/')), key=os.path.getmtime)
    if latestFolder:
        return latestFolder
    else:
        print(parentFolder, latestFolder)
        raise Exception("Train the neural network before testing!")