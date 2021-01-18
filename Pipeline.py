# Author: Kevin Wang
# Last Update: January 13, 2021

# Function: Pipeline to 1. preprocess data 2. reduce dimensions via VAE 3. test classification accuracy 4. Cross validate 5. Tune Parameters

# Outputs: 1 csv file: this csv file should depict accuracy for different parameters


#################################################################################

import preprocess_v2
import VAE_Train
import pandas as pd
from keras import backend as K


######## FOR GPU #################################
# Comment these lines if using
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

################### Pipeline #####################

print('Importing Dataset')
rawData = pd.read_csv('Lung_Features_3D_bk.csv')

survival, Normalized_features = preprocess_v2.preprocess(rawData)

Int_dim = [100,200,350,700] 
Lat_dim = [2,5,10,25,50]
Bat_size = [50,100,200,128]
L_R = [0.0005,0.001,0.002,0.01]


PT_results = []

for Idim in Int_dim:
	for Ldim in Lat_dim:
		for Bs in Bat_size:
			for Learn in L_R:
				sum_ = [0,0]
				av = [0,0]
				for fold in range(1,11):
					
					try:
						encoded_X_train, encoded_X_test, x_train, x_test, y_train, y_test = VAE_Train.dim_red(Idim,Ldim,Bs,Learn,Normalized_features,survival)

						accuracy = VAE_Train.test_model(encoded_X_train, encoded_X_test, y_train, y_test)

						print("Fold: " + str(fold))
						print("Accuracy: " + str(accuracy))
						print("")
						print("--------------------------")
						print("")

						
						sum_[0] = sum_[0] + accuracy[0]
						sum_[1] = sum_[1] + accuracy[1]

					except:
						print("Error in classifier")

					K.clear_session()
				try:
					av[0] = sum_[0]/10
					av[1] = sum_[1]/10	
					PT_results = PT_results + [[Idim,Ldim,Bs,Learn, av[0], av[1]]]
				except:
					PT_results = PT_results + [[Idim,Ldim,Bs,Learn,"Error"]]

toexcel = pd.DataFrame(PT_results)
toexcel.to_csv("PT_results_testing.csv")

###########################################################################################
####################################### Update Log ########################################

# January 13, 2021
# csv output for mist supercomputer compatibility

# January 4, 2021
# Updated inputs to test_model

# January 1, 2021
# Added parameter tuning and cross validation

# December 31, 2020
# File Created