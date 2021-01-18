import numpy as np
import pandas as pd


from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def reg_test(A, features, predictions):

	x_train, x_test, y_train, y_test = train_test_split(features, predictions, test_size=0.2, random_state=4)

	las = Lasso(alpha=  A, max_iter = 100000)
	las.fit(x_train, y_train)
	las_pred = las.predict(x_test)
	for i in range(0, len(las_pred)):
		if las_pred[i]>0.5:
			las_pred[i]=1
		else:
			las_pred[i]=0
	las_acc = accuracy_score(y_test, las_pred)

	rid = Lasso(alpha=  A, max_iter = 100000)
	rid.fit(x_train, y_train)
	rid_pred = rid.predict(x_test)
	for i in range(0, len(rid_pred)):
		if rid_pred[i]>0.5:
			rid_pred[i]=1
		else:
			rid_pred[i]=0
	rid_acc = accuracy_score(y_test, rid_pred)

	elastic = Lasso(alpha=  A, max_iter = 100000)
	elastic.fit(x_train, y_train)
	elastic_pred = elastic.predict(x_test)
	for i in range(0, len(elastic_pred)):
		if elastic_pred[i]>0.5:
			elastic_pred[i]=1
		else:
			elastic_pred[i]=0
	elastic_acc = accuracy_score(y_test, elastic_pred)

	output = [las_acc, rid_acc, elastic_acc, las_pred, rid_pred, elastic_pred]

	return output