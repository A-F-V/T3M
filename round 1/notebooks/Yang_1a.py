'Main problem: I removed empty data rows instead of estimating them, hence I only have c.330 predictions instead of the desired 400+'


import os,sys

globalT3Mbin = os.path.abspath(os.path.join(os.getcwd(),'..','..','bin'))
round1bin = os.path.abspath(os.path.join(os.getcwd(),'..','bin'))
if not (globalT3Mbin in sys.path):
    sys.path.append(globalT3Mbin)
if not (round1bin in sys.path):
    sys.path.append(round1bin)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, Perceptron
from sklearn import svm
from sklearn.neural_network import MLPClassifier


from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from utilities import splitter


'The below path would obviously only work on my computer, so running this code unedited probably isnt a good idea'
from pathlib import Path
data_folder_path = Path("C:/Users/yangyang/Documents/GitHub/T3M/round 1/data")
training_set_path = data_folder_path / "train.csv"




seed = 1
plt.style.use('dark_background')
from sklearn import set_config
#set_config(display='diagram')



data_set = pd.read_csv(training_set_path)
feature_names = data_set.columns.values
label_names = ['Survived']
feature_names= np.delete(feature_names, [3,8,10,11])            #gets rid of unnecessary columns in dataset
x_names = np.delete(feature_names, [1])                         #x_names is the list of columns which test_x will eventually take values from; this does not include 'Survived' ([1])
#print("The names of the features are - " +str(feature_names))



testing_set_path = data_folder_path / 'test.csv'
test_set = pd.read_csv(testing_set_path)
#print(feature_names)



#Converting male/female data into numbers so algorithms can understand this
for i in range (len(data_set['Sex'])):
    if data_set['Sex'][i] == 'male':
        data_set['Sex'][i] = 1
    else:
        data_set['Sex'][i] = 0

for i in range (len(test_set['Sex'])):
    if test_set['Sex'][i] == 'male':
        test_set['Sex'][i] = 1
    else:
        test_set['Sex'][i] = 0



test_x = test_set[x_names]
test_x = test_x.dropna(axis=0)      #gets rid of incomplete rows in test set
'ISSUE: the model should be able to predict values for incomplete rows; currently it cannot do this'



data_set = data_set[feature_names]
data_set = data_set.dropna(axis=0)  #gets rid of incomplete rows in dataset

train_x, train_y = data_set[x_names], data_set['Survived']  #splits dataset into x and y
#train_x = train_x.dropna(axis=0)    
#train_y = train_y.dropna(axis=0)


#randomforest

round1a_rf = RandomForestClassifier(random_state = 1)
round1a_rf.fit(train_x,train_y)
a = round1a_rf.predict(test_x)

#ridgeclassification
round1a_rc = RidgeClassifier(random_state=1)
round1a_rc.fit(train_x,train_y)
b = round1a_rc.predict(test_x)

#svm
round1a_svm = svm.SVC(random_state=1)
round1a_svm.fit(train_x, train_y)
c = round1a_svm.predict(test_x)


#MLPclassifier
round1a_mlp = MLPClassifier(random_state=1)
round1a_mlp.fit(train_x, train_y)
d = round1a_mlp.predict(test_x)

#Perceptron
round1a_perc = Perceptron(random_state=1)
round1a_perc.fit(train_x, train_y)
e = round1a_perc.predict(test_x)
#print(e)

#voting
vote = (((a+b+c+d+e)/5) > .5).astype(int)
print(len(vote), '\n', type(vote))
votedf = pd.DataFrame(data=vote, columns = ['Survived'])
votedf.to_excel('predictions.xlsx')

