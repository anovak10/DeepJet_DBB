# coding: utf-8

import sys
import os
import keras
import tensorflow as tf

from keras.losses import kullback_leibler_divergence, categorical_crossentropy
from keras.models import load_model, Model
from testing import testDescriptor
from argparse import ArgumentParser
from keras import backend as K
from Losses import * #needed!                                                                                                                
from modelTools import fixLayersContaining,printLayerInfosAndWeights
                                                                                                                                  
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from root_numpy import array2root
import pandas as pd
import h5py

#keras.backend.set_image_data_format('channels_last')

class MyClass:
    """A simple example class"""
    def __init__(self):
        self.inputDataCollection = ''
        self.outputDir = ''

import setGPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from training_base import training_base
from Losses import loss_NLL
import sys

trainDataCollection_pf_cpf_sv = '/cms-sc17/convert_20170717_ak8_deepDoubleB_db_pf_cpf_sv_train_val/dataCollection.dc'
trainDataCollection_cpf_sv = '/cms-sc17/convert_20170717_ak8_deepDoubleB_db_cpf_sv_train_val/dataCollection.dc'
trainDataCollection_sv='/cms-sc17/convert_20170717_ak8_deepDoubleB_db_sv_train_val/dataCollection.dc'

trainDataCollection_final = '/cms-sc17/convert_20170717_ak8_deepDoubleB_db_cpf_sv_reduced_train_val/dataCollection.dc'
trainDataCollection_final ='/afs/cern.ch/work/a/anovak/public/Jan19_train_full/dataCollection.dc'
testDataCollection_pf_cpf_sv = trainDataCollection_pf_cpf_sv.replace("train_val","test")
testDataCollection_cpf_sv = trainDataCollection_cpf_sv.replace("train_val","test")
testDataCollection_sv = trainDataCollection_sv.replace("train_val","test")

testDataCollection_final = trainDataCollection_final.replace("train","test")

sampleDatasets_pf_cpf_sv = ["db","pf","cpf","sv"]
sampleDatasets_cpf_sv = ["db","cpf","sv"]
sampleDatasets_sv = ["db","sv"]

#removedVars = [[],range(0,22),[0,1,2,3,4,5,6,7,8,9,10,13]]
removedVars = None

#Toggle training or eval
TrainBool = True
EvalBool = False

#Toggle to load model directly (True) or load weights (False)
LoadModel = False

#select model and eval functions
from DeepJet_models_final import conv_model_final as trainingModel
from eval_funcs import loadModel, makeRoc, _byteify, makeLossPlot, makeComparisonPlots
trainDir = "gpu_train_finalTest"
inputTrainDataCollection = trainDataCollection_final
inputTestDataCollection = testDataCollection_final
inputDataset = sampleDatasets_pf_cpf_sv
lossFunction = 'categorical_crossentropy'



if TrainBool:
    args = MyClass()
    args.inputDataCollection = inputTrainDataCollection
    args.outputDir = trainDir

    #also does all the parsing
    train=training_base(testrun=False,args=args)

    if not train.modelSet():

        train.setModel(trainingModel,inputDataset,removedVars)
    
        train.compileModel(learningrate=0.001,
                           loss=[lossFunction],
                           metrics=['accuracy'],
                           loss_weights=[1.])
    
        model,history,callbacks = train.trainModel(nepochs=1, 
                                                   batchsize=1024, 
                                                   stop_patience=1000, 
                                                   lr_factor=0.7, 
                                                   lr_patience=10, 
                                                   lr_epsilon=0.00000001, 
                                                   lr_cooldown=2, 
                                                   lr_minimum=0.00000001, 
                                                   maxqsize=100)

        train.keras_model=fixLayersContaining(train.keras_model, 'input_batchnorm')
        #printLayerInfosAndWeights(train.keras_model)
        model,history,callbacks = train.trainModel(nepochs=100,
                                                   batchsize=1024,
                                                   stop_patience=1000,
                                                   lr_factor=0.7,
                                                   lr_patience=10,
                                                   lr_epsilon=0.00000001,
                                                   lr_cooldown=2,
                                                   lr_minimum=0.00000001,
                                                   maxqsize=100)

if EvalBool:

    evalModel = loadModel(trainDir,inputTrainDataCollection,trainingModel,LoadModel,inputDataset,removedVars)
    evalDir = trainDir.replace('train','out')
    
    from DataCollection import DataCollection
    testd=DataCollection()
    testd.readFromFile(inputTestDataCollection)

    if os.path.isdir(evalDir):
        raise Exception('output directory: %s must not exists yet' %evalDir)
    else:
        os.mkdir(evalDir)

    df, features_val = makeRoc(testd, evalModel, evalDir)

    makeLossPlot(trainDir,evalDir)
    
