# coding: utf-8

import sys
import os
#import keras
#keras.backend.set_image_data_format('channels_last')

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

# Make all preformance plots
def make_plots(test_files_dc, model, outputDir):

    ## # summarize history for loss for training and test sample
    ## plt.figure(1)
    ## plt.plot(callbacks.history.history['loss'])
    ## plt.plot(callbacks.history.history['val_loss'])
    ## plt.title('model loss')
    ## plt.ylabel('loss')
    ## plt.xlabel('epoch')
    ## plt.legend(['train', 'test'], loc='upper left')
    ## plt.savefig(self.outputDir+'learningcurve.pdf') 
    ## plt.close(1)

    ## plt.figure(2)
    ## plt.plot(callbacks.history.history['acc'])
    ## plt.plot(callbacks.history.history['val_acc'])
    ## plt.title('model accuracy')
    ## plt.ylabel('acc')
    ## plt.xlabel('epoch')
    ## plt.legend(['train', 'test'], loc='upper left')
    ## plt.savefig(self.outputDir+'accuracycurve.pdf')
    ## plt.close(2)
    
    # Get variables from the test files data collection
    NENT = 10000000
    features_val = [fval[:NENT] for fval in test_files_dc.getAllFeatures()]
    labels_val=test_files_dc.getAllLabels()[0][:NENT,:]
    weights_val=test_files_dc.getAllWeights()[0][:NENT]
    spectators_val = test_files_dc.getAllSpectators()[0][:NENT,0,:]
    print features_val
    print labels_val
    print spectators_val
    df = pd.DataFrame(spectators_val)
    df.columns = ['fj_pt',
                  'fj_eta',
                  'fj_sdmass',
                  'fj_n_sdsubjets',
                  'fj_doubleb',
                  'fj_tau21',
                  'fj_tau32',
                  'npv',
                  'npfcands',
                  'ntracks',
                  'nsv']
    print(df.iloc[:10])

    # Run a prediction based on the trained model
    predict_test = model.predict(features_val)
    df['fj_isH'] = labels_val[:,1]
    df['fj_deepdoubleb'] = predict_test[:,1]
    # Cuts
    df = df[(df.fj_sdmass > 40) & (df.fj_sdmass < 200) & (df.fj_pt > 300) &  (df.fj_pt < 2500)]

    print(df.iloc[:10])

    # ROC curve from sklearn metrics
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
    # Double B
    fpr, tpr, threshold = roc_curve(df['fj_isH'],df['fj_deepdoubleb'])
    # Boosted decision trees - benchmark
    dfpr, dtpr, threshold1 = roc_curve(df['fj_isH'],df['fj_doubleb'])
    print threshold

    def find_nearest(array,value):
        idx = (np.abs(array-value)).argmin()
        return idx, array[idx]


    deepdoublebcuts = {}
    for wp in [0.01, 0.05, 0.1, 0.25, 0.5]: # % mistag rate
        idx, val = find_nearest(fpr, wp)
        deepdoublebcuts[str(wp)] = threshold[idx] # threshold for deep double-b corresponding to ~1% mistag rate
        print('deep double-b > %f coresponds to %f%% QCD mistag rate'%(deepdoublebcuts[str(wp)] ,100*val))

    # auc = computer area under a curve, auc(xs, ys)
    auc1 = auc(fpr, tpr)
    auc2 = auc(dfpr, dtpr)

    # Compare ROC curves for deep learnening and BDT
    plt.figure()       
    plt.plot(tpr,fpr,label='deep double-b, auc = %.1f%%'%(auc1*100))
    plt.plot(dtpr,dfpr,label='BDT double-b, auc = %.1f%%'%(auc2*100))
    plt.semilogy()
    plt.xlabel("H(bb) efficiency")
    plt.ylabel("QCD mistag rate")
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend()
    plt.savefig(outputDir+"test.pdf")
    
    # BDT discriminator plot
    plt.figure()
    bins = np.linspace(-1,1,70)
    plt.hist(df['fj_doubleb'], bins=bins, weights = 1-df['fj_isH'],alpha=0.5,normed=True,label='QCD')
    plt.hist(df['fj_doubleb'], bins=bins, weights = df['fj_isH'],alpha=0.5,normed=True,label='H(bb)')
    plt.xlabel("BDT double-b")
    plt.legend(loc='upper left')
    plt.savefig(outputDir+"doubleb.pdf")
    
    # Deep discriminator plot
    plt.figure()
    bins = np.linspace(0,1,70)
    plt.hist(df['fj_deepdoubleb'], bins=bins, weights = 1-df['fj_isH'],alpha=0.5,normed=True,label='QCD')
    plt.hist(df['fj_deepdoubleb'], bins=bins, weights = df['fj_isH'],alpha=0.5,normed=True,label='H(bb)')
    plt.xlabel("deep double-b")
    #plt.ylim(0.00001,1)
    #plt.semilogy()
    plt.legend(loc='upper left')
    plt.savefig(outputDir+"deepdoubleb.pdf")
    
    # Momentum
    plt.figure()
    bins = np.linspace(0,2000,70)
    plt.hist(df['fj_pt'], bins=bins, weights = 1-df['fj_isH'],alpha=0.5,normed=True,label='QCD')
    plt.hist(df['fj_pt'], bins=bins, weights = df['fj_isH'],alpha=0.5,normed=True,label='H(bb)')
    plt.xlabel(r'$p_{\mathrm{T}}$')
    plt.legend(loc='upper left')
    plt.savefig(outputDir+"pt.pdf")
    
    # Soft drop mass
    plt.figure()
    bins = np.linspace(40,200,41)
    plt.hist(df['fj_sdmass'], bins=bins, weights = 1-df['fj_isH'],alpha=0.5,normed=True,label='QCD')
    plt.hist(df['fj_sdmass'], bins=bins, weights = df['fj_isH'],alpha=0.5,normed=True,label='H(bb)')
    plt.xlabel(r'$m_{\mathrm{SD}}$')
    plt.legend(loc='upper left')
    plt.savefig(outputDir+"msd.pdf")
    
    # Soft drop mass for tagged H jets
    plt.figure()
    bins = np.linspace(40,200,41)
    df_passdoubleb = df[df.fj_doubleb > 0.9]
    plt.hist(df_passdoubleb['fj_sdmass'], bins=bins, weights = 1-df_passdoubleb['fj_isH'],alpha=0.5,normed=True,label='QCD')
    plt.hist(df_passdoubleb['fj_sdmass'], bins=bins, weights = df_passdoubleb['fj_isH'],alpha=0.5,normed=True,label='H(bb)')
    plt.xlabel(r'$m_{\mathrm{SD}}$')
    plt.legend(loc='upper left')
    plt.savefig(outputDir+"msd_passdoubleb.pdf")
    
    # Mass sculpting plots
    plt.figure()
    bins = np.linspace(40,200,41)
    for wp, deepdoublebcut in reversed(sorted(deepdoublebcuts.iteritems())):
        df_passdeepdoubleb = df[df.fj_deepdoubleb > deepdoublebcut]
        plt.hist(df_passdeepdoubleb['fj_sdmass'], bins=bins, weights = 1-df_passdeepdoubleb['fj_isH'],alpha=0.5,normed=True,label='QCD %i%% mis-tag'%(float(wp)*100.))
        #plt.hist(df_passdeepdoubleb['fj_sdmass'], bins=bins, weights = df_passdeepdoubleb['fj_isH'],alpha=0.5,normed=True,label='H(bb) %s'%wp)
    plt.xlabel(r'$m_{\mathrm{SD}}$')
    plt.legend(loc='upper right')
    plt.savefig(outputDir+"msd_passdeepdoubleb.pdf")
    
    plt.figure()
    bins = np.linspace(40,200,41)
    plt.hist(df['fj_sdmass'], bins=bins, weights = 1-df['fj_deepdoubleb'],alpha=0.5,normed=True,label='pred. QCD')
    plt.hist(df['fj_sdmass'], bins=bins, weights = df['fj_deepdoubleb'],alpha=0.5,normed=True,label='pred. H(bb)')
    plt.xlabel(r'$m_{\mathrm{SD}}$')
    plt.legend(loc='upper left')
    plt.savefig(outputDir+"msd_weightdeepdoubleb.pdf")

    return df

os.environ['CUDA_VISIBLE_DEVICES'] = ''

###############
#########
#################
# Set lcoations
inputDir = 'out_train_simple/'
inputModel = '%s/KERAS_check_best_model.h5'%inputDir
outputDir = inputDir.replace('train','eval') 

# test data:
inputDataCollection = '/eos/user/a/anovak/DeepJet/convertFromRoot/test_simple/dataCollection.dc'

# Check if dir works
if os.path.isdir(outputDir):
    raise Exception('output directory must not exists yet')
else: 
    os.mkdir(outputDir)

# Load model
model=load_model(inputModel, custom_objects=global_loss_list)
#intermediate_output = intermediate_layer_model.predict(data)
#print(model.summary())
    
# Load data    
from DataCollection import DataCollection
    
test_files_dc=DataCollection()
test_files_dc.readFromFile(inputDataCollection)

# Make plots    
df = make_plots(test_files_dc, model, outputDir)


##############################################
##############################################
############################################
# Make loss plots
# let's use only first 10000000 entries
NENT = 10000000
features_val = [fval[:NENT] for fval in test_files_dc.getAllFeatures()]

print model.summary()

def _byteify(data, ignore_dicts = False):
    # if this is a unicode string, return its string representation
    if isinstance(data, unicode):
        return data.encode('utf-8')
    # if this is a list of values, return list of byteified values
    if isinstance(data, list):
        return [ _byteify(item, ignore_dicts=True) for item in data ]
    # if this is a dictionary, return dictionary of byteified keys and values
    # but only if we haven't already byteified it
    if isinstance(data, dict) and not ignore_dicts:
        return {
            _byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True)
            for key, value in data.iteritems()
        }
    # if it's anything else, return it in its original form
    return data


import json
inputLogs = '%s/full_info.log'%inputDir
f = open(inputLogs)
myListOfDicts = json.load(f, object_hook=_byteify)
myDictOfLists = {}
for key, val in myListOfDicts[0].iteritems():
    myDictOfLists[key] = []
for i, myDict in enumerate(myListOfDicts):
    for key, val in myDict.iteritems():
        myDictOfLists[key].append(myDict[key])
val_loss = np.asarray(myDictOfLists['val_loss'])
loss = np.asarray(myDictOfLists['loss'])
plt.figure()
plt.plot(val_loss, label='validation')
plt.plot(loss, label='train')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig("%sloss.pdf"%outputDir)



