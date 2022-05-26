from get_data import get_data
import numpy as np


def add(i, isTrain):
    Data = []
    xTrain, yTrain = get_data(i, isTrain, 'dataset/')
    File_name = str(i)
    if isTrain:
        File_name += '-train.npy'
    else:
        File_name += '-test.npy'

    xTrain = np.load(File_name)


    for j in range(len(xTrain)):
        Train = xTrain[j]
        Train = np.append(Train, yTrain[j])
        Data.append(Train)
    
    File_name = str(i) + '-'
    if (isTrain):
        File_name += 'train'
    else:
        File_name += 'test'

    File_name += '.csv'
    np.savetxt(File_name, Data, delimiter=",")
    

for j in range(1, 10):
    add(j, 0)
    add(j, 1)