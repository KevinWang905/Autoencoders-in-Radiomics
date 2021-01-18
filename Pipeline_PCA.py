# Author: Kevin Wang
# Last Update: January 13, 2021

# Function: Pipeline to 1. preprocess data 2. reduce dimensions via PCA 3. test classification accuracy 4. Cross validate - future update 5. Tune Parameters

# Outputs: 1 excel file: this excel file should depict accuracy for different parameters


#################################################################################

import preprocess_v2
import VAE_Train
import pandas as pd
from keras import backend as K

################### Pipeline #####################

print('Importing Dataset')
rawData = pd.read_csv('Lung_Features_3D_bk.csv')

survival, Normalized_features = preprocess_v2.preprocess(rawData)


Components = [2,5,10,25,50]
PT_results = []

for comp in Components:

	sum_ = [0,0]
	av = [0,0]
	for fold in range(1,11):
		try:
			PCx_train, PCx_test, y_train, y_test = VAE_Train.PCA_red(Normalized_features,survival, comp)

			accuracy = VAE_Train.test_model(PCx_train, PCx_test,  y_train, y_test)

			print("Fold: " + str(fold))
			print("Accuracy: " + str(accuracy))
			print("")
			print("--------------------------")
			print("")


			sum_[0] = sum_[0] + accuracy[0]
			sum_[1] = sum_[1] + accuracy[1]

		except:
			print("Error1")

		K.clear_session()
	try:
		av[0] = sum_[0]/10
		av[1] = sum_[1]/10		
		PT_results = PT_results + [[comp, av[0],av[1]]]
	except:
		PT_results = PT_results + [[comp,"Error2"]]

toexcel = pd.DataFrame(PT_results)
toexcel.to_csv("PT_PCA_results.csv")

###########################################################################################
####################################### Update Log ########################################

# January 13, 2021
# Labels changed to binarized survival data.

# January 4, 2021
# File Created

