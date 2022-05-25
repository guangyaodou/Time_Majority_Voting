import sys
import time
import logging
from config import config, create_folder
from utils import IOHelper
from benchmark import benchmark
from utils.tables_utils import print_table
from get_data import get_data
import numpy as np


trainX, trainY = IOHelper.get_npz_data(config['data_dir'], verbose = True)


Data = []

for j in range(len(trainX)):
    cur = trainX[j][0].tolist()
    ID = trainY[j][0]
    label = trainY[j][1]

    cur.append(label)
    cur.append(ID)
    Data.append(cur)

np.savetxt("EEGEyeNet-data.csv", Data, delimiter=",")



