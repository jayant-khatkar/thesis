import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
import functools
import seaborn as sn



def plot_conf(conf):
    plt.figure(figsize = (10,7))
    sn.heatmap(conf, annot=False)
    plt.show()

def plot_hist(hist):
    plt.figure()
    hist = hist.item()
    train = hist['loss']
    val   = hist['val_loss']
    plt.plot(train)
    plt.plot(val)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plot_fine_hist(hist1, hist2, hist3):
    plt.figure()
    hist1 = hist1.item()
    hist2 = hist2.item()
    hist3 = hist3.item()
    train = hist1['loss'] + hist2['loss'] +hist3['loss']
    val   = hist1['val_loss'] + hist2['val_loss'] + hist3['val_loss']
    plt.plot(train)
    plt.plot(val)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
