import os
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
import pickle
from PIL import Image

def load(path='artifacts/data.pkl'):
    
    data = pd.read_csv('./data/english.csv')
    processed_data={}
    for i in [chr(i) for i in list(range(48, 58)) + list(range(65, 91)) + list(range(97, 123))]:
        processed_data[i]=np.empty((55,28,28))
        for j,k in zip(range(55),data[data['label'] == i]['image']):
            processed_data[i][j]=1-transforms.ToTensor()(Image.open('data/'+k).resize((28,28)))[0]
    
    with open(path, 'wb') as f:
        pickle.dump(processed_data, f)

if __name__=="__main__":
    load()