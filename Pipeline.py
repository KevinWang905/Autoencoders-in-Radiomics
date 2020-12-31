# Author: Kevin Wang
# Last Update: December 31, 2020

# Function: Pipeline to 1. preprocess data 2. reduce dimensions 3. test classification accuracy

# Outputs: 1 excel file: this excel file should encode for accuracy


#################################################################################

import preprocess_v2
import VAE_Train
import pandas as pd

################### Pipeline #####################

print('Importing Dataset')
rawData = pd.read_csv('Lung_Features_3D_bk.csv')

TStage, NStage, overallStage, Normalized_features = preprocess_v2.preprocess(rawData)

encoded_X_train, encoded_X_test, x_train, x_test, y_train, y_test = VAE_Train.dim_red(700,2,100,0.001,Normalized_features,NStage)

accuracy = VAE_Train.test_model(encoded_X_train, encoded_X_test, x_train, x_test, y_train, y_test)

print(accuracy)

###########################################################################################
####################################### Update Log ########################################

# December 31, 2020
# File Created