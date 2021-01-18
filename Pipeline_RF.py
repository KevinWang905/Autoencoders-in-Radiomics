# Author: Kevin Wang
# Last Update: January 16, 2021

# Function: Pipeline to 1. preprocess data 2. reduce dimensions via VAE 3. test RF classification accuracy 4. Cross validate 5. Tune Parameters for RF

# Outputs: 1 csv file: this excel file should depict accuracy for different parameters


#################################################################################

import preprocess_v2
import VAE_Train
import pandas as pd
import numpy as np
from keras import backend as K

################### Pipeline #####################

print('Importing Dataset')
rawData = pd.read_csv('Lung_Features_3D_bk.csv')

survival, Normalized_features = preprocess_v2.preprocess(rawData)

parameters = pd.read_csv('Params.csv')
param = []
for i in range(0,len(parameters.index)):
	param = param + [parameters.loc[i].to_numpy()]

n_estimators = [int(x) for x in np.linspace(start = 2, stop = 100, num = 5)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 5)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]





PT_results = []

for i in range(0,10):
	C_val = []
	sum_ = 0
	for fold in range(1,11):
		

		encoded_X_train, encoded_X_test, x_train, x_test, y_train, y_test = VAE_Train.dim_red(int(param[i][0]),int(param[i][1]),int(param[i][2]),float(param[i][3]),Normalized_features,survival)
		for e in n_estimators:
			for f in max_features:
				for d in max_depth:
					for s in min_samples_split:
						for l in min_samples_leaf:
							for b in bootstrap:
								print(e,f,d,s,l,b)
								accuracy = VAE_Train.test_RFmodel(encoded_X_train, encoded_X_test, y_train, y_test, e,f,d,s,l,b)
								print(accuracy)
								parameter_list = parameter_list = str(param[i][0])+ ", " +str(param[i][1]) + ", " + str(param[i][2]) + ", " + str(param[i][3])
								print(parameter_list)
								try:
									PT_results = PT_results + [[parameter_list, e,f,d,s,l,b, accuracy]]
								except:
									PT_results = PT_results + [[parameter_list,"Error1"]]

		print("Fold: " + str(fold))
		print("")
		print("--------------------------")
		print("")



		K.clear_session()
	

toexcel = pd.DataFrame(PT_results)
toexcel.to_csv("PT_RF_results.csv")

###########################################################################################
####################################### Update Log ########################################

# January 16, 2021
# added cross validation

# January 4, 2021
# File Created
