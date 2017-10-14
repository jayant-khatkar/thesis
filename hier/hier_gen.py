import pandas as pd
import numpy as np
import os
import random
import glob
from scipy.misc import imread, imresize, imshow
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator


#data_dir   = "/Users/user1/Documents/thesis/data/split_images/benthoz_retrain/training"
IM_WIDTH = 150

#/project/BEN_DL/benthoz2015/imagelist.csv
#/Users/user1/Documents/thesis/data/benthoz2015/imagelist.csv


#im_id = 1744579

hier_map = np.array(
    [[0,0,0],
    [0,1,1],
    [0,1,2],
    [0,1,3],
    [0,2,4],
    [0,3,5],
    [0,3,6],
    [0,3,7],
    [0,3,8],
    [0,3,9],
    [0,3,10],
    [0,3,10],
    [0,3,11],
    [0,4,12],
    [0,4,13],
    [0,4,13],
    [1,5,14],
    [1,5,15],
    [1,5,16],
    [1,6,17],
    [2,7,18],
    [0,3,6],
    [2,7,18],
    [0,0,0]]
    )

def hier_gen(data_dir, batch_size):

    datagen =  ImageDataGenerator(
        #preprocessing_function=preprocess_input,
        #rotation_range=10,
        horizontal_flip=True,
        vertical_flip=True,
        )

    generator = datagen.flow_from_directory(
        data_dir,
        target_size=(IM_WIDTH, IM_WIDTH),
        batch_size=batch_size,
    )

    while (True):
        ims, y_hot = next(generator)

        # Get hierarchcical data hierarchcical data here
        y = np.argmax(y_hot, axis =1)
        ys = hier_map[y] #converted to hierarchy
        b_size = len(y)
        y0 = np.zeros((b_size, 3), dtype=np.int8)
        y0[np.arange(b_size), ys[:,0]] = 1

        y1 = np.zeros((b_size, 8), dtype=np.int8)
        y1[np.arange(b_size), ys[:,1]] = 1

        y2 = np.zeros((b_size, 19), dtype=np.int8)
        y2[np.arange(b_size), ys[:,2]] = 1

        yield ims, [y0,y1,y2]
