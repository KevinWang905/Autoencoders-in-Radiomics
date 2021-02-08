# Author: Kevin Wang
# Last Update: February 5, 2021

# Function: 1. Uses VAE as dimensionality reduction using user inputed parameters then tunes gaussian SVC via various C and gamma values.

# Inputs: VAE parameters and GSVC parameters

# Outputs: CSV file depticting results of SVC tuning
#################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

import preprocess_v2
import VAE_Train


print('Importing Dataset')
rawData = pd.read_csv('Lung_Features_3D_bk.csv')

survival, Normalized_features = preprocess_v2.preprocess(rawData)

encoded_X_train, encoded_X_test, x_train, x_test, y_train, y_test = VAE_Train.dim_red(700,2,100,0.0001,Normalized_features,survival)
X = np.concatenate((encoded_X_train, encoded_X_test), axis=0) 
y = np.concatenate((y_train, y_test), axis=0) 
xtrain_enc = pd.DataFrame(X)
xtrain_enc.to_csv("X_temp.csv")

print("VAE Trained")

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)

PT_results = []

for i in C_range:
    for j in gamma_range:
        GSVC = SVC(kernel = 'rbf', C = i, gamma = j)
        GSVC.fit(encoded_X_train, y_train)
        GSVC_pred = GSVC.predict(encoded_X_test)

        GSVC_acc = accuracy_score(y_test, GSVC_pred)

        PT_results = PT_results + [[i, j, GSVC_acc]]

toexcel = pd.DataFrame(PT_results)
toexcel.to_csv("GSVC_PT_results.csv")