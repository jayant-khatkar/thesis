import pandas as pd
import numpy as np
import os
import random
import glob
from scipy.misc import imread, imresize, imshow
import matplotlib.pyplot as plt

#data_dir   = "/Users/user1/Documents/thesis/data/split_images/benthoz_retrain/training"
source_resize = 200

#/project/BEN_DL/benthoz2015/imagelist.csv
#/Users/user1/Documents/thesis/data/benthoz2015/imagelist.csv
imlist = pd.read_csv("/project/BEN_DL/benthoz2015/imagelist.csv")

#im_id = 1744579

def dual_im_gen(data_dir, batch_size):


    crop_addresses = glob.glob(os.path.join(data_dir,'*/*.jpg'))

    n_images = len(crop_addresses)
    if n_images==0:
        print("No images found")

    k=0
    while 1:

        # shuffle data
        crop_addresses = random.sample(crop_addresses, len(crop_addresses))
        labels = []
        for crop in crop_addresses:
            labels.append(int(crop.split('/')[-2]))

        #generate batches
        for i in range(int(n_images/batch_size)):
            batch_crops  = []
            batch_source = []
            batch_labels = []

            for j in range(batch_size):

                crop, source = get_im_pair(crop_addresses[k])

                batch_crops.append(crop)
                batch_source.append(source)
                batch_labels.append(labels[k])


                k=k+1
                if k==len(crop_addresses):
                    k=0
            b_label_out = np.zeros((batch_size, max(labels)+1), dtype=np.int8)
            b_label_out[np.arange(batch_size), batch_labels] = 1

            yield [np.array(batch_crops), np.array(batch_source)], b_label_out

def get_im_pair(crop_address):

    im_id = crop_address.split('/')[-1].split('-')[0]

    web_loc = imlist.loc[imlist['image__id']==im_id]['web_location']
    web_loc = web_loc[web_loc.index[0]]

    crop = imread(crop_address)
    source = get_source(web_loc)
    source = imresize(source, (source_resize,source_resize))

    #needed for inception V3 input
    crop = imresize(crop, (150,150))


    return crop, source

def get_source(web_loc):
    #/Users/user1/Documents/thesis/data/benthoz2015
    #/project/BEN_DL/benthoz2015/
    benthoz_dir = "/project/BEN_DL/benthoz2015"
    real_address = os.path.join(
        benthoz_dir,
        web_loc.split('/')[0],
        web_loc.split('/')[-1]
        )

    return imread(real_address)
