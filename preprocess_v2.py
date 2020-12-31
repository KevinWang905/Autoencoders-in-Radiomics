# Author: Kevin Wang
# Last Update: December 31, 2020

# Function: Preprocesses lung cancer radiomic data set 
#			removes exraneous info, binarizes survival time, normalizes features

# Inputs: 1 Dataframe: raw data 

# Outputs: 1 Dataframe: Normalized_features <- feature set
#		   3 Lists: NStage, TStage, OverallStage <- predictions
#								     


#################################################################################

import pandas as pd
import numpy as np

def binarize(time, event, threshold):

	if time < threshold:
		if event == 1:
			return 1
		else:
			return 0
	else:
		return 0

threshold = 500

def preprocess(lungData):


	print('One hot encoding of Gender')
	OH_prep = lungData.loc[:,['Gender']]
	OH = pd.get_dummies(OH_prep)
	lungData = pd.concat([lungData, OH], axis=1, ignore_index=False)

	print('Binarizing Survival Time and Deadstatus')
	result = [binarize(x, y, threshold) for x, y in zip(lungData['Survival.time'], lungData['deadstatus.event'])] 
	lungData['binarized'] = result


	print('Dropping Irrelevant columns and rows with missing data')
	lungData.drop(lungData.columns[11:22],axis = 1,  inplace = True)
	lungData.drop(lungData.columns[14:22],axis = 1,  inplace = True)
	lungData.drop(columns = ['Histology','PatientName', 'LesionID','Gender', 'Survival.time', 'deadstatus.event'], inplace = True)
	print('Dataset before dropping nans = '+str(lungData.shape))
	lungData.dropna(axis = 0,inplace= True)
	print('Dataset after dropping nans = '+str(lungData.shape))

	#print(lungData.iloc[0:5,1:5])
	print('Splitting Dataset into True Values and Featuress')
	features = lungData.copy()
	print(features.iloc[0:5,0:7])
	features = features.drop(features.columns[1:5], axis = 1)
	print("Shape of Features: " + str(features.shape))
	TStage = lungData.iloc[:,1]
	#TStage.to_excel("TStage.xlsx")

	NStage = lungData.iloc[:,2]
	#NStage.to_excel("NStage.xlsx")

	overallStage = lungData.iloc[:,4]
	overallStage = [1 if i == 'I' else 2 if i == 'II' else 3 if i == 'IIIa' else 4 for i in overallStage]
	toexcel = pd.DataFrame(overallStage)
	#toexcel.to_excel("overallStage.xlsx")

	print('Normalizing Dataframe')
	featuresNorm = (features-features.min())/(features.max()-features.min())
	featuresNorm.dropna(axis='columns', inplace = True)
	#featuresNorm.to_excel("Normalized_features.xlsx")

	return TStage, NStage, overallStage, featuresNorm


###########################################################################################
####################################### Update Log ########################################

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