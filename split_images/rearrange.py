import shutil
import pandas as pd
import os
import numpy as np



file_path1= "/project/BEN_DL/split_images/benthoz_split/test/labels_1000.csv"
labels = pd.read_csv(file_path1)

top_cats = [2,13,16,28,33,39,45,54,59,64,66,67,71,126,127,137,240,241,245,253,273,400,655,0]

for i in labels['label_id'].unique(): 
    directory = "/project/BEN_DL/split_images/benthoz_retrain/training/" + str(i)
    if  not os.path.isdir(directory):
        os.makedirs(directory)
    directory = "/project/BEN_DL/split_images/benthoz_retrain/testing/" + str(i)
    if  not os.path.isdir(directory):
        os.makedirs(directory)

#shutil.move("newfile.txt", "semester8/newfile.txt")
folder_from = "/project/BEN_DL/split_images/benthoz_split/test/"
folder_to   = "/project/BEN_DL/split_images/benthoz_retrain/testing/"


for i in labels['label_id'].unique():
    anns = labels.loc[labels['label_id'] == i]
    n=np.min([1000, len(anns)])
    anns = anns.sample(n)
    anns = anns.reset_index()
    for j in range(n):
        im_from = folder_from + anns['filename'][j].split('/')[-1] + '.jpg'
        im_to   = folder_to   + str(i)+'/' + anns['filename'][j].split('/')[-1].split('.')[0] + '.jpg'

        shutil.copy2(im_from, im_to)

    print(i)

