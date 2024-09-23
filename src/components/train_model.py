import os
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import pickle

class Discriminator(nn.Module):
    def __init__(self, dim = 32):
        super().__init__()
        self.dim = dim
        self.conv = nn.Sequential(
            nn.Conv2d(1, dim, 4, 2, 1),
            #nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Conv2d(dim, dim*2, 3, 1, 1),
            nn.BatchNorm2d(dim*2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Conv2d(dim*2, dim*4, 3, 1, 1),
            nn.BatchNorm2d(dim*4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Conv2d(dim*4, dim*8, 4, 2, 1),
            nn.BatchNorm2d(dim*8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        self.fc = nn.Linear(dim*8*7*7, 1)
    
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv(x)
        x = x.view(-1, self.dim*8*7*7)
        x = self.fc(x)
        
        #x = torch.sigmoid(x)
        
        return x.view(-1)
    

class Generator(nn.Module):
    def __init__(self, dim = 32, zdim = 100):
        super().__init__()
        self.dim = dim
        self.fc = nn.Linear(zdim, dim*8*7*7)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(dim*8, dim*4, 4, 2, 1),
            nn.BatchNorm2d(dim*4),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.ConvTranspose2d(dim*4, dim*2, 3, 1, 1),
            nn.BatchNorm2d(dim*2),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.ConvTranspose2d(dim*2, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.ConvTranspose2d(dim, 1, 4, 2, 1),
            #nn.BatchNorm2d(1),
            #nn.ReLU(),
            #nn.Dropout(0.2),
            
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.dim*8, 7, 7)
        x = self.conv(x)
        
        return x  
    
def train(data, batchsize=-1, epochs=-1, iters=1000, zdim=100):
    if batchsize==-1 or epochs==-1:
        batches=np.sqrt(len(data))
        batchsize=int(batches)
        epochs=1+int(iters/batches)
    device="cuda"
    start=timer()
    dataloader = DataLoader(data, batch_size = batchsize, shuffle = True)

    dis = Discriminator().to(device)
    gen = Generator().to(device)
    Loss = nn.BCEWithLogitsLoss()
    dis_optimizer = optim.Adam(dis.parameters(), lr = 0.0002, betas = (0.5, 0.999))
    gen_optimizer = optim.Adam(gen.parameters(), lr = 0.0002, betas = (0.5, 0.999))

    
    dis_loss = np.zeros(epochs)
    gen_loss = np.zeros(epochs)

    fixed_samples = torch.randn(9, zdim)
    fixed_samples = fixed_samples.to(device)
    print("preprocessing time =", timer()-start)
    
    for epoch in range(epochs):

        for x in dataloader:
            dis.train()
            gen.train()
            noise = torch.randn(x.shape[0], zdim).to(device).float()

            dis_optimizer.zero_grad()
            y_real = dis(x.to(device).float())
            fake_imgs = gen(noise).detach()
            y_fake = dis(fake_imgs.float())

            loss = Loss(y_real, torch.ones(x.shape[0]).to(device)) + Loss(y_fake, torch.zeros(x.shape[0]).to(device))
            dis_loss[epoch] += loss.item()
            loss.backward()
            dis_optimizer.step()  


            gen_optimizer.zero_grad()
            fake_imgs = gen(noise)
            y = dis(fake_imgs.float())

            loss = Loss(y, torch.ones(x.shape[0]).to(device))
            gen_loss[epoch] += loss.item()
            loss.backward()
            gen_optimizer.step()
        
        dis_loss[epoch]/=len(dataloader)
        gen_loss[epoch]/=len(dataloader)
        
        print("Epoch", epoch, "time =", timer()-start, 
        "Dis loss =", dis_loss[epoch],
        "Gen loss =", gen_loss[epoch])  
    gen.eval()
    gen.to("cpu")
    return gen

if __name__=="__main__":
    gen={}
    with open('artifacts/data.pkl', 'rb') as f:
        data = pickle.load(f)
    for i in data:
        gen[i] = train(data[i])
    with open('artifacts/model.pkl', 'wb') as f:
        pickle.dump(gen, f)
