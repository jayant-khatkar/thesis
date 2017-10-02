import numpy as np
import pandas as pd
from scipy import misc
import os
import time
import tensorflow as tf
import glob


#throwaway/testing/deprecated function
def image_loader_gen(folder_path, batch_size):

    images = glob.glob(folder_path+'/*.png')

    b=0

    while b+batch_size<len(images):

        loaded_images = []

        for i in range(b,b+batch_size):

            loaded_images.append(misc.imread(images[i]))

        yield loaded_images

        b+=batch_size

def batch_image_generator(folder_path, batch_size, n_epochs=1):
    #converts labels to one-hot for you

    labels = pd.read_csv(folder_path+'/labels_1000.csv')

    sess = tf.Session()
    for epoch in range(n_epochs):

        image_batch = []
        label_batch = []
        random_indices = np.random.permutation(len(labels))

        for im in range(len(random_indices)):
            im_path = folder_path +'/' +labels['filename'][random_indices[im]].split('/')[-1] + '.jpg'
            if os.path.isfile(im_path):
                image_batch.append(misc.imread(im_path))
                label_batch.append(labels['label_id'][random_indices[im]])

            if im % batch_size == batch_size - 1:
                #tf_label = tf.one_hot(label_batch, 274)
                #label_batch = sess.run(tf_label)
                yield image_batch, label_batch
                image_batch = []
                label_batch = []
