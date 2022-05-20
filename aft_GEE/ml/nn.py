#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import joblib
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from osgeo import gdal, gdalconst, ogr



class MyDataset(Dataset):
    def __init__(self, file_name, feature):
        self.file_name = file_name
        self.feature = feature
#         self.transforms = transforms
        data = pd.read_csv(self.file_name)
        data = data.dropna(axis = 0)
        self.x = data[self.feature].values
        self.y = data['label'].values
        self.len = len(data)
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        x_input = self.x[idx]
        label = self.y[idx] - 1
        return x_input, label


# In[22]:


def get_loader(feature, train_path, test_path):
    train_data = MyDataset(train_path, feature)
    test_data = MyDataset(test_path, feature)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    return train_loader, test_loader


# In[30]:


class Net(nn.Module):
    def __init__(self, nfeature):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(nfeature, 500)
        self.lin2 = nn.Linear(500, 1000)
        self.lin3 = nn.Linear(1000, 300)
        self.lin4 = nn.Linear(300, 2)
        self.bn1 = nn.BatchNorm1d(nfeature)
        self.bn2 = nn.BatchNorm1d(300)
        self.dropout = nn.Dropout(p=0.5)
#         self.bn2 = nn.BatchNorm1d(500)
#         self.bn3 = nn.BatchNorm1d(300)
        
    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = F.relu(self.lin2(x))
        x = self.dropout(x)
        x = F.relu(self.lin3(x))
        x = self.bn2(x)
        x = self.lin4(x)
        x = torch.sigmoid(x)
        return x


# In[40]:


def train(train_loader, model, optimizer, loss_func):
    model.train()
    losses = []
    TOTAL_EPOCHS = 50
    for epoch in range(TOTAL_EPOCHS):
        train_loss = 0.
        train_acc = 0.
        for i, (x, y) in enumerate(train_loader):
            x = x.float().to(device)
            y = y.long().to(device)
#             print(x.shape)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_func(outputs, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().data.item())
        if (epoch + 1) % 10 == 0:
            print(("epochs:{0}/{1}, loss:{2}".format(epoch, TOTAL_EPOCHS, np.mean(losses))))


# In[41]:


def test(test_loader, model, optimizer, loss_func):
    model.eval()
    correct = 0
    total = 0
    conf = np.zeros((2, 2))
    for i,(x, y) in enumerate(test_loader):
        x = x.float().to(device)
        y = y.long()
        outputs = model(x).cpu()
    #     print(outputs)
        _, predicted = torch.max(outputs.data, 1)
    #     print(predicted)
    #     print(y)
        total += y.size(0)
        correct += (predicted == y).sum()
        conf = conf + confusion_matrix(y, predicted)
    return conf


# In[11]:


def get_result(conf):
    print(conf)
    recall = conf[0][0] / conf[0].sum() * 100
    precision = conf[0][0] / conf[:, 0].sum() * 100
    accuracy = (conf[0][0] + conf[1][1]) / conf.sum() * 100
    F1 = 2 * recall * precision / (recall + precision)
    print('recall : %.4f %%' % recall)
    print('precision : %.4f %%' % precision)
    print('accuracy : %.4f %%' % accuracy)
    print('F1 : %.4f' % F1)
    print([recall, precision, accuracy, F1])
    return [recall, precision, accuracy, F1]




device = torch.device("cpu")
if torch.cuda.is_available():
        device = torch.device("cuda")




class Nn:
    def __init__(self, feature_list, train_path, test_path):
        print('nn')
        self.feature_list = feature_list
        self.train_path = train_path
        self.test_path = test_path
    
    
    def nn_result(self):
        result_ABC = []
        for feature in self.feature_list:
            nfeature = len(feature)
            train_loader, test_loader = get_loader(feature, self.train_path, self.test_path)
            model = Net(nfeature).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_func = torch.nn.CrossEntropyLoss()
            train(train_loader, model, optimizer, loss_func)
            conf = test(test_loader, model, optimizer, loss_func)
            result = get_result(conf)
            result_ABC.append(result)
        return result_ABC




