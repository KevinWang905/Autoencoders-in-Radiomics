# Author: Kevin Wang
# Last Update: February 1, 2021

# Function: 1. Wrangles data outputed from Pipeline_PI.py

# Inputs: N/A

# Outputs: Surface plot of specified classifer

#################################################################################

import pandas as pd
import numpy as np
rawData = pd.read_csv('PT_results.csv')

Latent_Dimensions = pd.DataFrame(rawData['1'].copy())
Learning_Rate = pd.DataFrame(rawData['3'].copy())
for i in range(0,len(Learning_Rate)):
    if Learning_Rate['3'][i] == 0.01:
        Learning_Rate['3'][i] = -2
    if Learning_Rate['3'][i] == 0.001:
        Learning_Rate['3'][i] = -3
    if Learning_Rate['3'][i] == 0.0001:
        Learning_Rate['3'][i] = -4
    if Learning_Rate['3'][i] == 0.00001:
        Learning_Rate['3'][i] = -5
    if Learning_Rate['3'][i] == 0.000001:
        Learning_Rate['3'][i] = -6

acc = rawData['4'].copy()
f1 = rawData['5'].copy()

for i in range (0,len(rawData['4'])):
    acc[i]= acc[i].replace("\n", "")
    acc[i]= acc[i].replace("[", "")
    acc[i]= acc[i].replace("]", "")
    acc[i] = [float(item) for item in acc[i].split()]
    acc[i] = np.array(acc[i])
    acc[i] = acc[i] * (5/4)
    acc[i]
    
for i in range (0,len(rawData['5'])):
    f1[i]= f1[i].replace("\n", "")
    f1[i]= f1[i].replace("[", "")
    f1[i]= f1[i].replace("]", "")
    f1[i] = [float(item) for item in f1[i].split()]
    f1[i] = np.array(f1[i])
    f1[i] = f1[i] * (5/4)
    f1[i]

KNN_acc, lr_acc, LSVC_acc, PSVC_acc, GSVC_acc, SSVC_acc, gnb_acc, RF_acc, DT_acc, NN_acc, ERF_acc = [],[],[],[],[],[],[],[],[],[],[]
for i in range(0, len(acc)):
    KNN_acc = KNN_acc + [acc[i][0]]
    lr_acc = lr_acc + [acc[i][1]]
    LSVC_acc = LSVC_acc + [acc[i][2]]
    PSVC_acc = PSVC_acc + [acc[i][3]]
    GSVC_acc = GSVC_acc + [acc[i][4]]
    SSVC_acc = SSVC_acc + [acc[i][5]]
    gnb_acc = gnb_acc + [acc[i][6]]
    RF_acc = RF_acc + [acc[i][7]]
    DT_acc = DT_acc + [acc[i][8]]
    NN_acc = NN_acc + [acc[i][9]]
    ERF_acc = ERF_acc + [acc[i][10]]

KNN_f1, lr_f1, LSVC_f1, PSVC_f1, GSVC_f1, SSVC_f1, gnb_f1, RF_f1, DT_f1, NN_f1, ERF_f1 = [],[],[],[],[],[],[],[],[],[],[]
for i in range(0, len(f1)):
    KNN_f1 = KNN_f1 + [f1[i][0]]
    lr_f1 = lr_f1 + [f1[i][1]]
    LSVC_f1 = LSVC_f1 + [f1[i][2]]
    PSVC_f1 = PSVC_f1 + [f1[i][3]]
    GSVC_f1 = GSVC_f1 + [f1[i][4]]
    SSVC_f1 = SSVC_f1 + [f1[i][5]]
    gnb_f1 = gnb_f1 + [f1[i][6]]
    RF_f1 = RF_f1 + [f1[i][7]]
    DT_f1 = DT_f1 + [f1[i][8]]
    NN_f1 = NN_f1 + [f1[i][9]]
    ERF_f1 = ERF_f1 + [f1[i][10]]

KNN_acc = pd.DataFrame(GSVC_acc)

knn_df = pd.concat([Latent_Dimensions['1'], Learning_Rate['3'], KNN_acc[0]], axis = 1)
knn_df.rename(columns={'1': "Latent_Dimensions", '3': "Learning_Rate", 0: "KNN_acc"}, inplace = True)

knn_contour = knn_df.groupby(['Latent_Dimensions','Learning_Rate']).mean()

grid_reset = knn_contour.reset_index()
grid_reset.columns = ['Latent_Dimensions', 'Learning_Rate', 'Accuracy']
grid_pivot = grid_reset.pivot('Latent_Dimensions', 'Learning_Rate')

x = grid_pivot.columns.levels[1].values
y = grid_pivot.index.values
z = grid_pivot.values


####### Plotly
import plotly.graph_objects as go
import plotly

fig = go.Figure(data=[go.Surface(z=z, y=y, x=x)])

fig.update_layout(title='GSVC Accuracy With VAE Hyperparameter Tuning', autosize=False,
                  width=600, height=600, scene = dict(
                    xaxis_title='Learning Rate (10^n)',
                    yaxis_title='Latent Dimensions',
                    zaxis_title='Accuracy'),
                  margin=dict(l=65, r=50, b=65, t=90))


plotly.offline.plot(fig)

