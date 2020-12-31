# Author: Kevin Wang
# Last Update: December 31, 2020

# Function: 1. Uses VAE as dimensionality reduction using user inputed parameters
#           2. Trains VAE using reduced data and determines accuracy

# Inputs: dim_red():
#         3 ints: intermediate dimensions, latent dimensions, batch size
#         1 float: learning rate
#         1 dataframe: feature matrix
#         1 list: predictions
#
#         test_model():
#         6 ndarrays: encoded_X_train, encoded_X_test, x_train, x_test, y_train, y_test

# Outputs: dim_red()
#          6 ndarrays: encoded_X_train, encoded_X_test, x_train, x_test, y_train, y_test
#
#          test_model():
#          1 float: KNN accuracy


#################################################################################
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import pandas as pd

import os
from scipy.stats import norm

from keras import backend as K
from keras import metrics, optimizers
from sklearn.model_selection import KFold
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.models import Model, Sequential
from keras.datasets import mnist
import lightgbm as lgb
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns

#Fixed Parameters

epsilon_std = 1.0
epochs = 200

################### Loss Functions ###########################

def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs



def dim_red(intermediate_dim, latent_dim, batch_size, learning_rate, features, predictions):

    # Trains autoencoder and uses it for dimensionality reduction

    # Variable Parameters
    print('Intermediate Dimensions: '+str(intermediate_dim))
    print('Latent Dimensions: '+str(latent_dim))
    print('Batch Size: '+str(batch_size))
    print('Learning Rate: '+str(learning_rate))

    print('Shape of features: ' + str(features.shape))
    original_dim = features.shape[1]
    features = features.to_numpy()
    predictions = np.array(predictions)
    
    ############# Model ##########################

    decoder = Sequential([
        Dense(intermediate_dim, input_dim=latent_dim, activation='relu'),
        Dense(original_dim, activation='sigmoid')
    ])

    x = Input(shape=(original_dim,))
    h = Dense(intermediate_dim, activation='relu')(x)

    z_mu = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
    z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)

    eps = Input(tensor=K.random_normal(stddev=epsilon_std,
                                       shape=(K.shape(x)[0], latent_dim)))
    z_eps = Multiply()([z_sigma, eps])
    z = Add()([z_mu, z_eps])

    x_pred = decoder(z)

    vae = Model(inputs=[x, eps], outputs=x_pred)
    adam = optimizers.Adam(lr=learning_rate)
    vae.compile(optimizer=adam, loss=nll)



    ########### Training Autoencoder ################

    x_train, x_test, y_train, y_test = train_test_split(features, predictions, test_size=0.2, random_state=4)
    print("xtrain")
    print(x_train)
    print("ytrain")
    print(y_train)


    vae.fit(x_train,
            x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, x_test))



    encoder = Model(x, z_mu)


    ############### Using Autoencoder as Dimensionality Reduction #############################


    encoded_X_train = encoder.predict(x_train)
    encoded_X_test = encoder.predict(x_test)


    ##################### Saving Train/Test Sets ##########################
    #uncomment for csv files of processed data


    # xtrain_enc = pd.DataFrame(encoded_X_train)
    # xtest_enc = pd.DataFrame(encoded_X_test)
    # xtrain = pd.DataFrame(x_train)
    # xtest = pd.DataFrame(x_test)
    # ytrain = pd.DataFrame(y_train)
    # ytest = pd.DataFrame(y_test)

    # xtrain_enc.to_csv("x_train_enc.csv")
    # xtest_enc.to_csv("x_test_enc.csv")
    # xtrain.to_csv("x_train_raw.csv")
    # xtest.to_csv("x_test_raw.csv")
    # ytrain.to_csv("y_train.csv")
    # ytest.to_csv("y_test.csv")
    #

    return encoded_X_train, encoded_X_test, x_train, x_test, y_train, y_test

def test_model(encoded_X_train, encoded_X_test, x_train, x_test, y_train, y_test):

    # Tests dimensionality reduced data using KNN

    model_name = "KNN"
    knnClass = KNN(n_neighbors = 4)

    knnClass.fit(encoded_X_train, y_train)
    knn_pred = knnClass.predict(encoded_X_test)
    knn_enc_acc = accuracy_score(y_test, knn_pred)

    return knn_enc_acc


###########################################################################################
####################################### Update Log ########################################

# December 31, 2020
# Added test_model function
# Changed outputs of dim_red from csv files to dataframes
# Changed original_dim variable to change based on results of preprocessing

# December 2020
# File created