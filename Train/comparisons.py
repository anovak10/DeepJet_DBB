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

testDataCollection_pf_cpf_sv = trainDataCollection_pf_cpf_sv.replace("train_val","test")
testDataCollection_cpf_sv = trainDataCollection_cpf_sv.replace("train_val","test")
testDataCollection_sv = trainDataCollection_sv.replace("train_val","test")

testDataCollection_final = trainDataCollection_final.replace("train_val","test")


sampleDatasets_pf_cpf_sv = ["db","pf","cpf","sv"]
sampleDatasets_cpf_sv = ["db","cpf","sv"]
sampleDatasets_sv = ["db","sv"]

removedVars = [[],range(0,22),[0,1,2,3,4,5,6,7,8,9,10,13]]

from DeepJet_models_removals import deep_model_removals as trainingModel
from eval_funcs import loadModel, makeRoc, _byteify, makeLossPlot, makeComparisonPlots

#for Comparisons
from DeepJet_models_removals import deep_model_removal_sv, deep_model_removals,conv_model_removals
from DeepJet_models_ResNet import deep_model_doubleb_sv
from DeepJet_models_final import conv_model_final
bins=[10,20,40]
batches = [1024,8192]

compareDir = "finalTestComparisons/"

compModels = [conv_model_removals,conv_model_removals,conv_model_final]
compNames = ["baseline","lambdaLayers","finalTest"]
compRemovals = [removedVars,removedVars,None]
compLoadModels = [False,False,False]
compTrainDirs = ["train_conv_db_cpf_sv_removals/","../independence/example/train_conv_db_cpf_sv_removals_lossfunc_nbins40_b8192","../independence/example/train_finalTest_nbins40_b8192"]
compDatasets = [sampleDatasets_cpf_sv,sampleDatasets_cpf_sv,sampleDatasets_cpf_sv]    
compTrainDataCollections = [trainDataCollection_cpf_sv,trainDataCollection_cpf_sv,trainDataCollection_final]
compTestDataCollections = [testDataCollection_cpf_sv,testDataCollection_cpf_sv,testDataCollection_final]

#for batch in batches:
    #for b in bins:
        #compModels.append(conv_model_removals)
        #compNames.append("%d bins; %d batch"%(b,batch))
        #compRemovals.append(removedVars)
        #compLoadModels.append(False)
        #compTrainDirs.append("../independence/example/train_conv_db_cpf_sv_removals_lossfunc_nbins%d_b%d"%(b,batch))
        #compDatasets.append(sampleDatasets_cpf_sv)
        #compTrainDataCollections.append(trainDataCollection_cpf_sv)
        #compTestDataCollections.append(testDataCollection_cpf_sv)

#compModels = [trainingModel, deep_model_removals, deep_model_removal_sv, deep_model_removal_sv, deep_model_doubleb_sv]
#compNames = ["trackVars","d3d+d3dsig","SV-ptrel_erel_pt_mass","SV-pt,e,etaRel_deltaR_pt_mass","SV"]
#compRemovals = (removedVars,[[],[0,1,2,3,4,5,6,7,8,9,10,13]],[0,1,5,6],[0,1,3,4,5,6],[])
#compLoadModels = [LoadModel,False,False,False,True]
#compTrainDirs = [trainDir,'train_deep_sv_removals_d3d_d3dsig_only/',"train_deep_sv_removals_ptrel_erel_pt_mass/","train_deep_sv_removals_ptrel_erel_etarel_deltaR_pt_mass/","train_deep_init_64_32_32_b1024/"]
#compareDir = "comparedROCSTrack/"
#compDatasets = [sampleDatasets_pf_cpf_sv,["db","sv"],["db","sv"],["db","sv"],["db","sv"]]
#compTrainDataCollections = [trainDataCollection_pf_cpf_sv,trainDataCollection_sv,trainDataCollection_sv,trainDataCollection_sv,trainDataCollection_sv]
#compTestDataCollections = [testDataCollection_pf_cpf_sv, testDataCollection_sv,testDataCollection_sv,testDataCollection_sv,testDataCollection_sv]

    
from DataCollection import DataCollection

if os.path.isdir(compareDir):
    raise Exception('output directory: %s must not exists yet' %compareDir)
else:
    os.mkdir(compareDir)
        


models = []
testds = []
for i in range(len(compModels)):
    curModel = loadModel(compTrainDirs[i],compTrainDataCollections[i],compModels[i],compLoadModels[i],compDatasets[i],compRemovals[i])
    testd = DataCollection()
    testd.readFromFile(compTestDataCollections[i])

    models.append(curModel)
    testds.append(testd)

makeComparisonPlots(testds,models,compNames,compareDir)
