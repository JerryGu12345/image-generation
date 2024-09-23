import os
from timeit import default_timer as timer
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import pickle
#from train_model import Generator, train
#from load_data import load
from src.components.train_model import Generator

def generate(input, gen, path="artifacts/"):
    rows = 15
    cols = 20
    fig, axes = plt.subplots(rows,cols)
    for i in range(rows*cols):
        ax=axes[i//cols,i%cols]
        if i<len(input) and input[i]!=' ':
            samples = gen[input[i]](torch.randn(1, 100).float())
            img = samples[0].detach()[0]
        else:
            img = torch.zeros(28,28)
        ax.imshow(img, cmap='Greys_r')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    plt.subplots_adjust(wspace=0, hspace=0)
    #plt.savefig("artifacts/fig"+str(timer())+".jpg")
    plt.savefig(path+"fig.jpg")

if __name__=="__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    with open('artifacts/model.pkl', 'rb') as f:
        gen = pickle.load(f)
    generate("01 01",gen,"artifacts/")
