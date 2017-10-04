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
    sn.heatmap(conf, annot=True)
    plt.show()
