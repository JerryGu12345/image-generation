import os
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
import pickle
from PIL import Image

def load(path='artifacts/data.pkl'):
    
    data = pd.read_csv('./data/mnist_train.csv')[0:6000]
    processed_data = {}
    for i in range(10):
        processed_data[str(i)]=torch.Tensor(data[data['label'] == i].values[:,1:].reshape(-1, 28, 28)/255)*2-1
    with open(path, 'wb') as f:
        pickle.dump(processed_data, f)

if __name__=="__main__":
    load()