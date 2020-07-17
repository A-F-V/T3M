from sklearn import set_config
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.linear_model import RidgeClassifier, Perceptron
from sklearn.ensemble import RandomForestClassifier
'Main problem: I removed empty data rows instead of estimating them, hence I only have c.330 predictions instead of the desired 400+'


import os
import sys

print(os.getcwd())
globalT3Mbin = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'bin'))
round1bin = os.path.abspath(os.path.join(os.getcwd(), '..', 'bin'))
if not (globalT3Mbin in sys.path):
    sys.path.append(globalT3Mbin)
if not (round1bin in sys.path):
    sys.path.append(round1bin)


#from utilities import splitter


'The below path would obviously only work on my computer, so running this code unedited probably isnt a good idea'
data_folder_path = os.path.join(os.getcwd(), 'round 1', 'data')
training_set_path = data_folder_path / "train.csv"


seed = 1
plt.style.use('dark_background')
# set_config(display='diagram')


'''
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
'''

# randomforest

round1a_rf = RandomForestClassifier(random_state=1)
round1a_rf.fit(training_set_X_trans, training_set_Y)
a = round1a_rf.predict(test_set_X_trans)

# ridgeclassification
round1a_rc = RidgeClassifier(random_state=1)
round1a_rc.fit(training_set_X_trans, training_set_Y)
b = round1a_rc.predict(test_set_X_trans)

# svm
round1a_svm = svm.SVC(random_state=1)
round1a_svm.fit(training_set_X_trans, training_set_Y)
c = round1a_svm.predict(test_set_X_trans)


# MLPclassifier
round1a_mlp = MLPClassifier(random_state=1)
round1a_mlp.fit(training_set_X_trans, training_set_Y)
d = round1a_mlp.predict(test_set_X_trans)

# Perceptron
round1a_perc = Perceptron(random_state=1)
round1a_perc.fit(training_set_X_trans, training_set_Y)
e = round1a_perc.predict(test_set_X_trans)
# print(e)

# voting
vote = (((a+b+c+d+e)/5) > .5).astype(int)
print(len(vote), '\n', type(vote))
votedf = pd.DataFrame(data=vote, columns=['Survived'])
votedf.to_excel('predictions.xlsx')


print(vote.score(test_set_X_trans, test_set_Y))
