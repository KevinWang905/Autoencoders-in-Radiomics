# Author: Kevin Wang
# Last Update: Feb 1, 2021

# Function: Pipeline to 1. preprocess data 2. reduce dimensions via VAE 3. test classification accuracy 4. Cross validate 5. Tune Parameters

# Outputs: 1 csv file: this csv file should depict accuracy and f1 scores for different parameters


#################################################################################

import preprocess_v2
import VAE_Train
import ModelTesting
import pandas as pd
import numpy as np
from keras import backend as K



################### Pipeline #####################

print('Importing Dataset')
rawData = pd.read_csv('Lung_Features_3D_bk.csv')

survival, Normalized_features = preprocess_v2.preprocess(rawData)

Int_dim = [700] 
Lat_dim = [2,4,6,8,10,50,100,200]
Bat_size = [100]
L_R = [0.01,0.001,0.0001,0.0001,0.00001,0.000001]



PT_results = []

for Idim in Int_dim:
	for Ldim in Lat_dim:
		for Bs in Bat_size:
			for Learn in L_R:
				acc = np.array([0,0,0,0,0,0,0,0,0,0,0])
				f1_ = np.array([0,0,0,0,0,0,0,0,0,0,0])
				p1_ = np.array([0,0,0,0,0,0,0,0,0,0])
				for fold in range(1,6):
					encoded_X_train, encoded_X_test, x_train, x_test, y_train, y_test = VAE_Train.dim_red(Idim,Ldim,Bs,Learn,Normalized_features,survival)
					# try:
					if (np.count_nonzero(np.isnan(encoded_X_train))==0):	

						accuracy, f1_scores, p1_scores = ModelTesting.test_model(Ldim,encoded_X_train, encoded_X_test, y_train, y_test)

						print("Fold: " + str(fold))
						print("Accuracy: " + str(accuracy))
						print("F1: " + str(f1_scores))
						print("Percent 1s: " + str(p1_scores))
						print("")
						print("--------------------------")
						print("")

						
						acc = acc + accuracy
						f1_ = f1_ + f1_scores
						for i in range(0,len(p1_)):
							if p1_scores[i] == 1:
								p1_[i] = p1_[i] + 1
					else:
						print("NANS!")
						acc = np.array([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])
						f1_ = np.array([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])
						p1_ = np.array([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan])


					# except:
					# 	print("Error in classifier")

					K.clear_session()
				try:
					av = acc/5
					f1_ = f1_/5	
					PT_results = PT_results + [[Idim,Ldim,Bs,Learn, av, f1_, p1_]]
				except:
					PT_results = PT_results + [[Idim,Ldim,Bs,Learn,"Error"]]

toexcel = pd.DataFrame(PT_results)
toexcel.to_csv("PT_results.csv")

###########################################################################################
####################################### Update Log ########################################

# Feb 1, 2021
# File Created

