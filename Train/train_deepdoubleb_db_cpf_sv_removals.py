
# coding: utf-8

# In[1]:

import sys
import os
import keras
#keras.backend.set_image_data_format('channels_last')



# In[2]:

class MyClass:
    """A simple example class"""
    def __init__(self):
        self.inputDataCollection = ''
        self.outputDir = ''


# In[5]:
import setGPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from training_base import training_base
from Losses import loss_NLL
import sys

args = MyClass()
args.inputDataCollection = '/cms-sc17/convert_20170717_ak8_deepDoubleB_db_cpf_sv_train_val/dataCollection.dc'
args.outputDir = 'train_conv_db_cpf_sv_removals'

sampleDatasets = ["db","cpf","sv"]
removedVars = [[],range(0,22),[0,1,2,3,4,5,6,7,8,9,10,13]]
#removedVars = None

TrainBool = True
EvalBool= True

train=training_base(testrun=False,args=args)


if TrainBool:
    #also does all the parsing

    if not train.modelSet():
        from DeepJet_models_removals import conv_model_removals as model

        train.setModel(model,sampleDatasets,removedVars)
    
        train.compileModel(learningrate=0.001,
                           loss=['categorical_crossentropy'],
                           metrics=['accuracy'])
    

        model,history,callbacks = train.trainModel(nepochs=300, 
                                                   batchsize=1024, 
                                                   stop_patience=1000, 
                                                   lr_factor=0.5, 
                                                   lr_patience=10, 
                                                   lr_epsilon=0.00000001, 
                                                   lr_cooldown=2, 
                                                   lr_minimum=0.00000001, 
                                                   maxqsize=100)


