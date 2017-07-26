

from training_base import training_base
#from Losses import loss_NLL

#also dows all the parsing
train=training_base(testrun=True)

from models import dense_model
from DeepJet_callbacks import DeepJet_callbacks

callbacks=DeepJet_callbacks(stop_patience=300,

                            lr_factor=0.5,
                            lr_patience=2,
                            lr_epsilon=0.003,
                            lr_cooldown=6,
                            lr_minimum=0.000001,

                            outputDir="test")

train.setModel(dense_model,dropoutRate=0.1)

train.compileModel(learningrate=0.005,
                   loss=['categorical_crossentropy'],
                   metrics=['accuracy'])


model,history = train.trainModel(nepochs=5, 
                                 batchsize=250, 
                                 stop_patience=300, 
                                 lr_factor=0.5, 
                                 lr_patience=10, 
                                 lr_epsilon=0.0001, 
                                 lr_cooldown=2, 
                                 lr_minimum=0.0001, 
                                 maxqsize=10)