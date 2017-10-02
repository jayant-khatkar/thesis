import numpy as np
import pandas as pd
import os
import sys

ann_folder = '/Users/user1/Desktop/thesis/data/annotations/'
ann_files  = [f for f in os.listdir(ann_folder) if f.endswith('.csv') and f.startswith('ann')]

anns = pd.DataFrame()
for FILE_NAME in ann_files:
   temp = pd.read_csv(ann_folder + FILE_NAME)
   anns = pd.concat([anns, temp]) 

print(len(anns))

names = anns['name']

all_cats = names.unique()

ubranches = []

for cat in all_cats:
    ubranches.append(cat.split(': '))

level1 = []
level2 = []
level3 = []
level4 = []
for branch in ubranches:
    level1.append(branch[0])
    if len(branch)>1:
        level2.append(branch[0]+': '+branch[1])
    if len(branch)>2:
        level3.append(branch[0]+': '+branch[1]+': '+branch[2])
    if len(branch)>3:
        level4.append(branch[0]+': '+branch[1]+': '+branch[2]+': '+branch[3])
level1 = list(set(level1))
level2 = list(set(level2))
level3 = list(set(level3))
level4 = list(set(level4))

lev1_counts = np.zeros(len(level1))
for i, cat in enumerate(level1):
    for sample in names:
        if sample.startswith(cat):
            lev1_counts[i] += 1

lev1_freq = np.concatenate((level1,lev1_counts)).reshape(2, -1).T

lev2_counts = np.zeros(len(level2))
for i, cat in enumerate(level2):
    for sample in names:
        if sample.startswith(cat):
            lev2_counts[i] += 1

lev2_freq = np.concatenate((level2,lev2_counts)).reshape(2, -1).T

lev3_counts = np.zeros(len(level3))
for i, cat in enumerate(level3):
    for sample in names:
        if sample.startswith(cat):
            lev3_counts[i] += 1

lev3_freq = np.concatenate((level3,lev3_counts)).reshape(2, -1).T

lev4_counts = np.zeros(len(level4))
for i, cat in enumerate(level4):
    for sample in names:
        if sample.startswith(cat):
            lev4_counts[i] += 1

lev4_freq = np.concatenate((level4,lev4_counts)).reshape(2, -1).T

