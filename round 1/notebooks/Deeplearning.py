import os
import sys
from pathlib import Path
data_folder_path = Path("round 1/data")
training_set_path = data_folder_path / "train.csv"
test_set_path = data_folder_path / "test.csv"
output_path = data_folder_path / "output.csv"
globalT3Mbin = os.path.abspath(os.path.join(os.getcwd(), 'bin'))
round1bin = os.path.abspath(os.path.join(os.getcwd(), 'bin'))

if not (globalT3Mbin in sys.path):
    sys.path.append(globalT3Mbin)
if not (round1bin in sys.path):
    sys.path.append(round1bin)

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC, SVC
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiplicativeLR, ExponentialLR
from sklearn import set_config
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from transformers.FeatureMapper import FeatureMapper
from sklearn.impute import SimpleImputer
from IPython.display import display
from IPython.display import Image
from scipy import linalg
from math import ceil
from transformers.FeatureSelector import FeatureSelector
from utilities import splitter
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import SGD
import math
from collections import OrderedDict

seed = 1
plt.style.use('dark_background')
set_config(display='diagram')


cat_selector = FeatureSelector(['Sex', 'Embarked', 'Pclass'])
# should pclass be numerical or categorical? - I think it should be categorical (NN)
num_selector = FeatureSelector(['Age', 'SibSp', 'Parch', 'Fare'])
cat_imputer = SimpleImputer(strategy='most_frequent')
num_imputer = SimpleImputer(strategy='mean')
cat_encoder = OneHotEncoder()
num_scalar = StandardScaler()


def get_deck(cabin):
    return str(cabin)[0]


cabin_selector = FeatureSelector(['Cabin'])
cabin_mapper = FeatureMapper(get_deck)
cabin_encoder = OneHotEncoder()

num_pipeline = Pipeline(
    [('selector', num_selector), ('imputer', num_imputer), ('scalar', num_scalar)])
cat_pipeline = Pipeline([('selector', cat_selector),
                         ('imputer', cat_imputer), ('encoder', cat_encoder)])
cabin_pipeline = Pipeline([('selector', cabin_selector),
                           ('mapper', cabin_mapper), ('encoder', cabin_encoder)])
full_pipeline = FeatureUnion(transformer_list=[('num_pipeline', num_pipeline), (
    'cat_pipeline', cat_pipeline), ('cabin_pipeline', cabin_pipeline)])

data_set = pd.read_csv(training_set_path)
feature_names = data_set.columns.values
label_names = ['Survived']


data_set_X, data_set_Y = splitter.split_from_labels(data_set, label_names)

training_set, validation_set, test_set = splitter.split_tvt_set(data_set, [
                                                                1, 0, 0], seed)
training_set_X, training_set_Y = splitter.split_from_labels(
    training_set, label_names)
test_set_X, test_set_Y = splitter.split_from_labels(test_set, label_names)

training_set_X_trans = full_pipeline.fit_transform(training_set_X)
#test_set_X_trans = full_pipeline.fit_transform(test_set_X)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
training_X_tensor = torch.from_numpy(
    training_set_X_trans.toarray()).float().to(device)
training_Y_tensor = torch.from_numpy(
    training_set_Y.to_numpy()).float().to(device)
#test_X_tensor = torch.from_numpy(test_set_X_trans.toarray()).float().to(device)
test_Y_tensor = torch.from_numpy(test_set_Y.to_numpy()).float().to(device)
trainDataSet = TensorDataset(training_X_tensor, training_Y_tensor)
trainDataLoader = DataLoader(
    trainDataSet, batch_size=16, drop_last=True)
#testDataSet = TensorDataset(test_X_tensor, test_Y_tensor)
#testDataLoader = DataLoader(testDataSet, batch_size=16, drop_last=True)


# class Net(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.l1 = nn.Linear(21,1)
#
#    def forward(self,x):
#        return F.sigmoid(self.l1(x))
#net = nn.Sequential(nn.Linear(21,21),nn.ReLU(),nn.Linear(21,1),nn.Sigmoid())

def train(trainloader, model, epochs, lr_init=1, gamma=0.99):
    #optimizer = SGD(model.parameters(), lr_init, momentum=0.7, nesterov=True)
    optimizer =  torch.optim.Adam(model.parameters(),lr_init,weight_decay=0.01)
    #scheduler = CosineAnnealingWarmRestarts(optimizer, t0,tmult)
    scheduler = ExponentialLR(optimizer,gamma=gamma)
    criterion = nn.BCEWithLogitsLoss()
    loses = []
    for epoch in range(epochs):
        running_loss = 0
        for i, data in enumerate(trainloader):
            x, y = data
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            running_loss += loss
            loss.backward()
            optimizer.step()
        loses.append(running_loss)
        scheduler.step()
        if epoch % 20 == 19:
            print("Epoch number: " + str(epoch+1) + " and Learning rate is: "+ str([param_group['lr'] for param_group in optimizer.param_groups])+ " and Loss of: " + str(running_loss.item()))
    return loses


def accuracy(testloader, model):
    count = 0
    total = len(testloader.dataset)
    for data in testloader:
        input, label = data[0], data[1]
        pred = model(input)
        count += ((pred >= 0.5).int() == label).sum().item()
    return count/total


def loss(testloader, model, criterion):
    loss = 0
    for data in testloader:
        x, y = data[0], data[1]
        pred = model(x)
        loss += criterion(pred, y)
    return loss


def debugTrainer(trainer, epochs=100):
    x = range(1, epochs+1)
    y = trainer(epochs)
    plt.plot(x, y)
    plt.show()

class ResBlock(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs

def createNet(layers,expansion,dropout_rate,activationFunc = nn.ReLU()):
    modules = [nn.Dropout(0.1),nn.BatchNorm1d(21), nn.Linear(21, 21*expansion), activationFunc]
    for i in range(layers-1):
        modules.append(nn.Dropout(p=dropout_rate))
        modules.append(nn.BatchNorm1d(21*expansion))
        modules.append(nn.Linear(21*expansion,21*expansion))
        modules.append(activationFunc)
    modules.append(nn.BatchNorm1d(21*expansion))
    modules.append(nn.Linear(21*expansion,1))
    modules.append(nn.Sigmoid())
    return nn.Sequential(OrderedDict(zip(map(str,range(len(modules))),modules)))

def createResNet(layers,dropout_rate,activationFunc = nn.ReLU()):
    modules = [ResBlock(nn.Sequential(nn.Dropout(0.1),nn.BatchNorm1d(21), nn.Linear(21,21))), activationFunc]
    for i in range(layers-1):
        modules.append(ResBlock(nn.Sequential(nn.Dropout(p=dropout_rate),nn.BatchNorm1d(21), nn.Linear(21,21))))
        modules.append(activationFunc)
    modules.append(nn.BatchNorm1d(21))
    modules.append(nn.Linear(21,1))
    modules.append(nn.Sigmoid())
    return nn.Sequential(OrderedDict(zip(map(str,range(len(modules))),modules)))


#net = createNet(20,5,0.3,nn.LeakyReLU()).to(device)
#debugTrainer((lambda epochs: train(trainDataLoader, net, epochs, 0.01)), 500)
#print(accuracy(trainDataLoader, net))
#print(accuracy(testDataLoader, net))
net = createResNet(5,0.1,nn.ReLU()).to(device)
debugTrainer((lambda epochs: train(trainDataLoader, net, epochs, 0.001,0.99)), 300)
print(accuracy(trainDataLoader, net))

final_test_set = pd.read_csv(test_set_path)
fdata = full_pipeline.transform(final_test_set)
def createPrediction(predictions):
    predictions = (predictions>=0.5).astype(int)
    ids = final_test_set["PassengerId"].to_numpy().reshape(-1,1)
    output_data = np.concatenate((ids,predictions),axis=1)
    df = pd.DataFrame(data=output_data,columns =["PassengerId","Survived"])
    df.to_csv(output_path,index=False)
fdata_tensor = torch.from_numpy(fdata.toarray()).float().to(device)
createPrediction(net(fdata_tensor).cpu().detach().numpy())