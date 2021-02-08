# Author: Kevin Wang
# Last Update: January 13, 2021

# Function: 1. Trains various classifiers and outputs accuracy, f1 score, and number of outputs that were entirely ones

# Inputs: latent dim, X_train, x_test, y_train, y_test

# Outputs: 3 np arrays

#################################################################################

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
## Classifiers

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

## Neural Net
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder

def fit_pred(model, X_train, y_train, X_test, y_test):
    
    model.fit(X_train, y_train)
    model_pred = model.predict(X_test)

    model_acc = accuracy_score(y_test, model_pred)
    model_acc = round(model_acc, 4)

    model_f1 = f1_score(y_test, model_pred)
    model_f1 = round(model_f1, 4)

    ones = np.count_nonzero(model_pred == 1)
    percent1 = ones/len(model_pred)
    percent1 = round(percent1, 4)

    return model_acc, model_f1, percent1

def ANN(I_Dims, X_train,X_test,y_train,y_test):


    model = Sequential()
    keras.layers.Flatten(input_shape=(I_Dims,))
    model.add(Dense(round((I_Dims+2)/2), activation='relu'))
    model.add(Dense(round((I_Dims+2)/2), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size = 1, verbose = False)

    y_pred = model.predict(X_test)



    for i in range(0,len(y_pred)):

        y_pred[i][0] = round(y_pred[i][0])


    model_acc = accuracy_score(y_test, y_pred)
    model_acc = round(model_acc, 4)
    model_f1 = f1_score(y_test, y_pred)
    model_f1 = round(model_f1, 4)



    return model_acc, model_f1


def test_model(lat_dim, encoded_X_train, encoded_X_test, y_train, y_test):

    #KNN
    knnClass = KNN(n_neighbors = 2)
    knn_enc_acc, knn_enc_f1, knn_enc_p1 = fit_pred(knnClass, encoded_X_train, y_train, encoded_X_test, y_test)

    #Logistic Regression
    lr=LogisticRegression(max_iter=1000)
    lr_acc, lr_f1, lr_p1 = fit_pred(lr, encoded_X_train, y_train, encoded_X_test, y_test)

    #Support Vector Classifiers
    #Linear
    LSVC = SVC(kernel = 'linear')
    LSVC_acc, LSVC_f1, LSVC_p1 = fit_pred(LSVC, encoded_X_train, y_train, encoded_X_test, y_test)
    #Polynomial (Default = 3)
    PSVC = SVC(kernel = 'poly')
    PSVC_acc, PSVC_f1, PSVC_p1 = fit_pred(PSVC, encoded_X_train, y_train, encoded_X_test, y_test)
    #Gaussian
    GSVC = SVC(kernel = 'rbf')
    GSVC_acc, GSVC_f1, GSVC_p1 = fit_pred(GSVC, encoded_X_train, y_train, encoded_X_test, y_test)
    #Sigmoid
    SSVC = SVC(kernel = 'sigmoid')
    SSVC_acc, SSVC_f1, SSVC_p1 = fit_pred(SSVC, encoded_X_train, y_train, encoded_X_test, y_test)

    #Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb_acc, gnb_f1, gnb_p1 = fit_pred(gnb, encoded_X_train, y_train, encoded_X_test, y_test)

    #Random Forest
    RF = RandomForestClassifier()
    RF_acc, RF_f1, RF_p1 = fit_pred(RF, encoded_X_train, y_train, encoded_X_test, y_test)

    #Random Forest
    ERF = ExtraTreesClassifier()
    ERF_acc, ERF_f1, ERF_p1 = fit_pred(ERF, encoded_X_train, y_train, encoded_X_test, y_test)

    #Decision Tree
    DT = DecisionTreeClassifier()
    DT_acc, DT_f1, DT_p1 = fit_pred(DT, encoded_X_train, y_train, encoded_X_test, y_test)

    #Neural Net
    NN_acc, NN_f1 = ANN(lat_dim, encoded_X_train, encoded_X_test, y_train, y_test)

    accuracies = np.array([knn_enc_acc, lr_acc, LSVC_acc, PSVC_acc, GSVC_acc, SSVC_acc, gnb_acc, RF_acc, DT_acc, NN_acc, ERF_acc])
    f1_scores = np.array([knn_enc_f1, lr_f1, LSVC_f1, PSVC_f1, GSVC_f1, SSVC_f1, gnb_f1, RF_f1, DT_f1, NN_f1, ERF_f1])
    p1_scores = np.array([knn_enc_p1, lr_p1, LSVC_p1, PSVC_p1, GSVC_p1, SSVC_p1, gnb_p1, RF_p1, DT_p1, ERF_p1])

    return accuracies, f1_scores, p1_scores

###########################################################################################
####################################### Update Log ########################################

# February 1, 2020
# File created