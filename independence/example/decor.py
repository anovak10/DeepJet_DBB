# Simple example to present a methods to add a measure of independence w.r.t. a variable "a" to the loss.
import numpy as np
import sys
import h5py
import setGPU
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Dropout
from keras.losses import kullback_leibler_divergence, categorical_crossentropy
from Losses import * # import NBINS from Losses
from keras import backend as K
from keras.callbacks import EarlyStopping
import tensorflow as tf
from DataCollection import DataCollection
import matplotlib.pyplot as plt
plt.switch_backend('agg')

seed = 7
np.random.seed(seed)

def main():
    
    inputDataCollection = '/cms-sc17/convert_20170717_ak8_deepDoubleB_db_cpf_sv_train_val/dataCollection.dc'
    
    traind=DataCollection()
    traind.readFromFile(inputDataCollection)

    
    NENT = 1 # take all events
    binWidth = (200.-40.)/NBINS
    features_val = [fval[::NENT] for fval in traind.getAllFeatures()] # get all files
    labels_val=traind.getAllLabels()[0][::NENT,:]
    spectators_val = traind.getAllSpectators()[0][::NENT,0,:]    
    #spath = traind.getSamplePath(traind.samples[0])
    #h5File = h5py.File(spath)
    #print h5File.items()
    #features_val = [h5File['x%i'%j][()] for j in range(0, h5File['x_listlength'][()][0])] # get only first file
    #labels_val = h5File['y0'][()] 
    #spectators_val = h5File['z0'][()][:,0,:]

    # OH will be the truth "y" input to the network
    # OH contains both, the actual truth per sample and the actual bin (one hot encoded) of the variable to be independent of
    OH = np.zeros((labels_val.shape[0],NBINS+2))
    print labels_val.shape
    print labels_val.shape[0]
    
    for i in range(0,labels_val.shape[0]):
        # bin of a (want to be independent of a)
        OH[i,int((spectators_val[i,2]-40.)/binWidth)]=1
        # aimed truth (target) 
        OH[i,NBINS] = labels_val[i,0]
        OH[i,NBINS+1] = labels_val[i,1]

    # make a simple model:
    from DeepJet_models_removals import conv_model_removals
    #fresh model:
    sampleDatasets = ["db","cpf","sv"]
    removedVars = [[],range(0,22),[0,1,2,3,4,5,6,7,8,9,10,13]]
    #model = conv_model_removals([Input(shape=(1,27,)),Input(shape=(60,30,)),Input(shape=(5,14,))], 2, 0, sampleDatasets, removedVars)
    model = conv_model_removals([Input(shape=(1,27,)),Input(shape=(60,30,)),Input(shape=(5,14,))], 2, NBINS, sampleDatasets, removedVars)
    # load weights from standard training:
    #from keras.models import load_model_weights
    
    # compile with custom loss
    model.compile(loss=loss_reg,optimizer='adam', metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    model.load_weights('../../Train/train_conv_db_cpf_sv_removals/KERAS_check_best_model_weights.h5',by_name=True)
    # batch size is huge because of need to evaluate independence


    from DeepJet_callbacks import DeepJet_callbacks
    
    callbacks=DeepJet_callbacks(stop_patience=1000,
                            lr_factor=0.5,
                            lr_patience=10,
                            lr_epsilon=0.000001,
                            lr_cooldown=2,
                            lr_minimum=0.0000001,
                            outputDir='train_conv_db_cpf_sv_removals_reg_pretrain/')
    model.fit(features_val, OH, batch_size=1024, epochs=200, 
              verbose=1, validation_split=0.2, shuffle = True, 
              callbacks = callbacks.callbacks)
    #model.fit(features_val, labels_val, batch_size=1024, epochs=200, 
    #          verbose=1, validation_split=0.2, shuffle = True, 
    #          callbacks = callbacks.callbacks)
    # get the truth:
    #output = model.predict(x)


if __name__ == "__main__":

    main()
    



