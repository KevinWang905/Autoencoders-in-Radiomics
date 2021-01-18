# Author: Kevin Wang
# Last Update: January 15, 2021

# Function: Pipeline to 1. preprocess data 2. reduce dimensions via lasso, ridge, and elasticnet 3. test classification accuracy 4. Cross validate 5. Tune Parameters

# Outputs: 1 csv file: this csv file should depict accuracy for different parameters


#################################################################################

import preprocess_v2
import LinReg_Regularized
import pandas as pd
import numpy as np


######## FOR GPU #################################
# Comment these lines if using CPU
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

################### Pipeline #####################

print('Importing Dataset')
rawData = pd.read_csv('Lung_Features_3D_bk.csv')

survival, Normalized_features = preprocess_v2.preprocess(rawData)
Alpha = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 2.0]


PT_results = []


for A in Alpha:

	Acc_list = np.array([0,0,0], dtype = 'f')

	for fold in range(1,11):
		
		try:
			fold_acc = LinReg_Regularized.reg_test(A, Normalized_features, survival)
			Acc_list = Acc_list + fold_acc[0:3]

			print("Fold: " + str(fold))
			print("Accuracy: " + str(fold_acc))
			print("Sum: " + str(Acc_list))
			print("")
			print("--------------------------")
			print("")


		except:
		 	print("Error in classifier")

	try:
		Av_acc = Acc_list/10	
		PT_results = PT_results + [[A, Av_acc[0], Av_acc[1], Av_acc[2]]]
	except:
		PT_results = PT_results + [[A,"Error"]]

toexcel = pd.DataFrame(PT_results)
toexcel.to_csv("reg_test_results.csv")

###########################################################################################
####################################### Update Log ########################################

# January 15, 2021
# File Created
