from collections import OrderedDict
import math
from torch.optim import SGD
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from bin.utilities import splitter
from bin.transformers.FeatureSelector import FeatureSelector
from math import ceil
from scipy import linalg
from IPython.display import Image
from IPython.display import display
from sklearn.impute import SimpleImputer
from bin.transformers.FeatureMapper import FeatureMapper
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import set_config
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiplicativeLR, ExponentialLR
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline, FeatureUnion
import os
import sys
from pathlib import Path
from torch.utils.data import random_split
import mlflow
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

print(data_set_X.shape)
training_set, validation_set, test_set = splitter.split_tvt_set(data_set, [
                                                                1, 0, 0], seed)
print(training_set.shape)
training_set_X, training_set_Y = splitter.split_from_labels(
    training_set, label_names)
test_set_X, test_set_Y = splitter.split_from_labels(test_set, label_names)

training_set_X_trans = full_pipeline.fit_transform(training_set_X)
# test_set_X_trans = full_pipeline.fit_transform(test_set_X)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
training_X_tensor = torch.from_numpy(
    training_set_X_trans.toarray()).float().to(device)
training_Y_tensor = torch.from_numpy(
    training_set_Y.to_numpy()).float().to(device)
# test_X_tensor = torch.from_numpy(test_set_X_trans.toarray()).float().to(device)
# test_Y_tensor = torch.from_numpy(test_set_Y.to_numpy()).float().to(device)
trainDataSet = TensorDataset(training_X_tensor, training_Y_tensor)
trainDataLoader = DataLoader(
    trainDataSet, batch_size=32, drop_last=True)
# testDataSet = TensorDataset(test_X_tensor, test_Y_tensor)
# testDataLoader = DataLoader(testDataSet, batch_size=32, drop_last=True)


# class Net(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.l1 = nn.Linear(21,1)
#
#    def forward(self,x):
#        return F.sigmoid(self.l1(x))
# net = nn.Sequential(nn.Linear(21,21),nn.ReLU(),nn.Linear(21,1),nn.Sigmoid())

def train(trainloader, model, epochs, lr_init=1, weight_decay=0.01, log=False):
   # optimizer = SGD(model.parameters(), lr_init, momentum=0.9, nesterov=True)
    optimizer = torch.optim.Adam(model.parameters(), lr_init, weight_decay=weight_decay)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, t0,tmult)
    #scheduler = ExponentialLR(optimizer, gamma=gamma)
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
       # scheduler.step()
        if(log):
            if epoch % 20 == 19:
                print("Epoch number: " + str(epoch+1) + " and Learning rate is: " +
                      str([param_group['lr'] for param_group in optimizer.param_groups]) + " and Loss of: " + str(running_loss.item()))
    return loses


def accuracy(testloader, model):
    count = 0
    total = len(testloader.dataset)
    for data in testloader:
        input, label = data[0], data[1]
        pred = model(input)
        count += ((pred >= 0).int() == label).sum().item()
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


def createNet(layers, expansion, dropout_rate, activationFunc=nn.ReLU()):
    modules = [nn.Dropout(0.1), nn.BatchNorm1d(21), nn.Linear(21, 21*expansion), activationFunc]
    for i in range(layers-1):
        modules.append(nn.Dropout(p=dropout_rate))
        modules.append(nn.BatchNorm1d(21*expansion))
        modules.append(nn.Linear(21*expansion, 21*expansion))
        modules.append(activationFunc)
    modules.append(nn.BatchNorm1d(21*expansion))
    modules.append(nn.Linear(21*expansion, 1))
    return nn.Sequential(*modules)


def createResNet(layers, dropout_rate, activationFunc=nn.ReLU()):
    modules = [ResBlock(nn.Sequential(nn.Dropout(0.1), nn.BatchNorm1d(21), nn.Linear(21, 21))), activationFunc]
    for i in range(layers-1):
        modules.append(ResBlock(nn.Sequential(nn.Dropout(p=dropout_rate), nn.BatchNorm1d(21), nn.Linear(21, 21))))
        modules.append(activationFunc)
    modules.append(nn.BatchNorm1d(21))
    modules.append(nn.Linear(21, 1))
    return nn.Sequential(*modules)


# net = createNet(4,2,0.3,nn.LeakyReLU()).to(device)
# debugTrainer((lambda epochs: train(trainDataLoader, net, epochs, 0.01)), 500)
# print(accuracy(trainDataLoader, net))
# print(accuracy(testDataLoader, net))


data_set_X, data_set_Y = splitter.split_from_labels(data_set, label_names)
data_set_X_trans = full_pipeline.fit_transform(data_set_X)
data_X_tensor = torch.from_numpy(
    data_set_X_trans.toarray()).float().to(device)
data_Y_tensor = torch.from_numpy(
    data_set_Y.to_numpy()).float().to(device)

ds = TensorDataset(data_X_tensor, data_Y_tensor)


# ARGS [0]-layers [1]-dropout_rate [2]-lr [3]-weight_decay [4] -epoch [5] -expansion

def permuteArgs(args):
    output = []
    for layer in args['layers']:
        for dr in args['dropout_rate']:
            for lr in args['lr']:
                for weight_decay in args['weight_decay']:
                    for epoch in args['epochs']:
                        for exp in args['expansion']:
                            output.append([layer, dr, lr, weight_decay, epoch, exp])
    return output


def cross_validate(model, dataSet, testMetric, args, splitSize, cvs, bs=16):
    bestscore_val = 0
    worstscore_val = 1
    bestscore_train = 0
    worstscore_train = 1
    for arg in permuteArgs(args):
        with mlflow.start_run():
            metric_cum_train = 0
            metric_cum_test = 0
            for c in range(cvs):
                net = model(arg[0], arg[5], arg[1]).to(device)
                tSet, vSet = random_split(
                    dataSet, [int(len(dataSet)*splitSize), len(dataSet)-int(len(dataSet)*splitSize)])
                train(DataLoader(tSet, batch_size=bs, drop_last=True), net, arg[4], arg[2], arg[3])
                met_train = testMetric(DataLoader(tSet, batch_size=bs, drop_last=True), net)
                metric_cum_train += met_train
                met_test = testMetric(DataLoader(vSet, batch_size=bs, drop_last=True), net)
                metric_cum_test += met_test
            aveVal = metric_cum_test/cvs
            aveTrain = metric_cum_train/cvs
            mlflow.log_param('layers', arg[0])
            mlflow.log_param('dr', arg[1])
            mlflow.log_param('lr', arg[2])
            mlflow.log_param('weight_decay', arg[3])
            mlflow.log_param('epochs', arg[4])
            mlflow.log_param('expansion', arg[5])
            mlflow.log_metric('VACC', aveVal)
            mlflow.log_metric('TACC', aveTrain)
            if cvs != 1:
                print(f"AVERAGE: {arg} - VAL:{aveVal} TRAIN:{aveTrain}")
            if bestscore_val < aveVal:
                best_val = arg
                bestscore_val = aveVal
                print(f"New best VALIDATION {arg} of score {aveVal}")
            if worstscore_val > aveVal:
                worst_val = arg
                worstscore_val = aveVal
                print(f"New worst VALIDATION {arg} of score {aveVal}")
            if bestscore_train < aveTrain:
                best_train = arg
                bestscore_train = aveTrain
                print(f"New best TRAIN {arg} of score {aveTrain}")
            if worstscore_train > aveTrain:
                worst_train = arg
                worstscore_train = aveTrain
                print(f"New worst TRAIN {arg} of score {aveTrain}")
    print(f"OVERALL BEST: VAL:{best_val} of score {bestscore_val} TRAIN:{best_train} of score {bestscore_train}")
    print(f"OVERALL WORST: VAL:{worst_val} of score {worstscore_val} TRAIN:{worst_train} of score {worstscore_train}")


#args = {"layers": [2], "dropout_rate": [0.02], "lr": [0.08, 0.05, 0.03, 0.01, 0.001], "lr_decay": [0.99, 0.98], "momentum": [0.95, 0.9], "epochs": [50, 100, 150]}
#args = {"layers": [2, 3], "dropout_rate": [ 0.01, 0.02, 0.03], "lr": [0.05, 0.03, 0.01], "lr_decay": [1, 0.995, 0.99], "momentum": [0.95, 0.9], "epochs": [50]}
#cross_validate(createResNet, ds, accuracy, args, 0.25, 5, 64)
args = {"layers": [2, 3, 5, 10], "dropout_rate": [0.02, 0.05, 0.1], "lr": [0.05, 0.005, 0.0005,
                                                                           0.00005], "weight_decay": [0.001, 0.01, 0.1],  "epochs": [50, 100, 200], "expansion": [1, 3, 5, 10]}
cross_validate(createNet, ds, accuracy, args, 0.25, 3, 64)

net = createResNet(2, 0.05, nn.LeakyReLU()).to(device)
#debugTrainer((lambda epochs: train(trainDataLoader, net, epochs, 0.05, 0.99)), 100)
dataLoader = DataLoader(ds, batch_size=32, drop_last=True)
train(dataLoader, net, 150, 0.03, 0.9, 0.99, True)
print(accuracy(dataLoader, net))

final_test_set = pd.read_csv(test_set_path)
fdata = full_pipeline.transform(final_test_set)


def createPrediction(predictions):
    predictions = (predictions >= 0).astype(int)
    ids = final_test_set["PassengerId"].to_numpy().reshape(-1, 1)
    output_data = np.concatenate((ids, predictions), axis=1)
    df = pd.DataFrame(data=output_data, columns=["PassengerId", "Survived"])
    df.to_csv(output_path, index=False)


fdata_tensor = torch.from_numpy(fdata.toarray()).float().to(device)
createPrediction(net(fdata_tensor).cpu().detach().numpy())
