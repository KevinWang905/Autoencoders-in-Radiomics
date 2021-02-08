# Author: Kevin Wang
# Last Update: January 31, 2021

# Function: Preprocesses lung cancer radiomic data set 
#			removes exraneous info, binarizes survival time, normalizes features

# Inputs: 1 Dataframe: raw data 

# Outputs: 1 Dataframe: Normalized_features <- feature set
#		   3 Lists: NStage, TStage, OverallStage <- predictions
#								     


#################################################################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

def binarize(time, event, threshold):

	if time < threshold:
		if event == 1:
			return 1
		else:
			return 0
	else:
		return 0

threshold = 730

def preprocess(lungData):



	print('Binarizing Survival Time and Deadstatus')
	result = [binarize(x, y, threshold) for x, y in zip(lungData['Survival.time'], lungData['deadstatus.event'])] 
	lungData['binarized'] = result


	print('Dropping Irrelevant columns and rows with missing data')
	lungData.drop(lungData.columns[0:33],axis = 1,  inplace = True)
	print('Dataset before dropping nans = '+str(lungData.shape))
	lungData.dropna(axis = 0,inplace= True)
	print('Dataset after dropping nans = '+str(lungData.shape))


	print('Splitting Dataset into Labels and Featuress')
	features = lungData.copy()

	survival = features['binarized']
	survival = survival.to_numpy()
	features.drop(columns = ['binarized'], inplace = True)

	

	print('Normalizing Dataframe')
	
	featuresNorm = features + abs(features.min())
	featuresNorm = (features-features.min())/(features.max()-features.min())
	featuresNorm.dropna(axis='columns', inplace = True)
	
	print("Shape of Features: " + str(features.shape))
##########################################################################################


########### Uncomment this block if excel files are needed



	# featuresNorm.to_excel("Normalized_features.xlsx") 
	# toexcel = survival
	# toexcel.to_excel("survival.xlsx")


###########################################################################################

	return survival, featuresNorm


###########################################################################################
####################################### Update Log ########################################

# January 31, 2021
# Threshold changed to 730 days, normalization changed to 0 to 1 (from -1 to 1).

# January 5, 2021
# Outputted feature list consists of only radiomic data. Outputted label changed to survival

# December 31, 2020
# Added step to remove nans after normalizing dataframe; columns with all same value would
#	produce a column of nans after normalization. This is beneficial for the model as well
#	as it removes columns of useless data
#
# Changed output. Instead returns 4 dataframes

# December 2020
# Added binarize function to binarize deadstatus and survival time

# November 2020
# File created