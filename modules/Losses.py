from keras import backend as K
from keras.losses import kullback_leibler_divergence, categorical_crossentropy
import tensorflow as tf

global_loss_list={}

#whenever a new loss function is created, please add it to the global_loss_list dictionary!
def loss_kldiv(y_in,x):
    # h is the histogram vector "one hot encoded" (40 bins in this case), techically part of the "truth" y                         
    h = y_in[:,0:40]
    y = y_in[:,40:]

    h_btag_anti_q = K.dot(K.transpose(h), K.dot(tf.diag(y[:,0]),x))
    h_btag_anti_h = K.dot(K.transpose(h), K.dot(tf.diag(y[:,1]),x))
    h_btag_q = h_btag_anti_q[:,1]
    h_btag_q = h_btag_q / K.sum(h_btag_q,axis=0)
    h_anti_q = h_btag_anti_q[:,0]
    h_anti_q = h_anti_q / K.sum(h_anti_q,axis=0)
    h_btag_h = h_btag_anti_h[:,1]
    h_btag_h = h_btag_h / K.sum(h_btag_h,axis=0)
    h_anti_h = h_btag_anti_q[:,0]
    h_anti_h = h_anti_h / K.sum(h_anti_h,axis=0)

    return categorical_crossentropy(y, x) + \
        kullback_leibler_divergence(h_anti_q, h_btag_q) + \
        kullback_leibler_divergence(h_btag_h, h_anti_h)

#please always register the loss function here                                                                                              
global_loss_list['loss_kldiv']=loss_kldiv

def loss_NLL(y_true, x):
    """
    This loss is the negative log likelyhood for gaussian pdf.
    See e.g. http://bayesiandeeplearning.org/papers/BDL_29.pdf for details
    Generally it might be better to even use Mixture density networks (i.e. more complex pdf than single gauss, see:
    https://publications.aston.ac.uk/373/1/NCRG_94_004.pdf
    """
    x_pred = x[:,1:]
    x_sig = x[:,:1]
    return K.mean(0.5* K.log(K.square(x_sig))  + K.square(x_pred - y_true)/K.square(x_sig)/2.,    axis=-1)

#please always register the loss function here
global_loss_list['loss_NLL']=loss_NLL

def loss_meansquared(y_true, x):
    """
    This loss is the negative log likelyhood for gaussian pdf.
    See e.g. http://bayesiandeeplearning.org/papers/BDL_29.pdf for details
    Generally it might be better to even use Mixture density networks (i.e. more complex pdf than single gauss, see:
    https://publications.aston.ac.uk/373/1/NCRG_94_004.pdf
    """
    x_pred = x[:,1:]
    x_sig = x[:,:1]
    return K.mean(0.5* K.square(x_sig)  + K.square(x_pred - y_true)/2.,    axis=-1)

#please always register the loss function here
global_loss_list['loss_meansquared']=loss_meansquared


# The below is to use multiple gaussians for regression

## https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation/blob/master/MDN-DNN-Regression.ipynb
## these next three functions are from Axel Brando and open source, but credits need be to https://creativecommons.org/licenses/by-sa/4.0/ in case we use it

def log_sum_exp(x, axis=None):
    """Log-sum-exp trick implementation"""
    x_max = K.max(x, axis=axis, keepdims=True)
    return K.log(K.sum(K.exp(x - x_max), 
                       axis=axis, keepdims=True))+x_max
                       

global_loss_list['log_sum_exp']=log_sum_exp

def mean_log_Gaussian_like(y_true, parameters):
    """Mean Log Gaussian Likelihood distribution
    Note: The 'c' variable is obtained as global variable
    """
    
    #Note: The output size will be (c + 2) * m = 6
    c = 1 #The number of outputs we want to predict
    m = 2 #The number of distributions we want to use in the mixture
    components = K.reshape(parameters,[-1, c + 2, m])
    mu = components[:, :c, :]
    sigma = components[:, c, :]
    alpha = components[:, c + 1, :]
    alpha = K.softmax(K.clip(alpha,1e-8,1.))
    
    exponent = K.log(alpha) - .5 * float(c) * K.log(2 * np.pi) \
    - float(c) * K.log(sigma) \
    - K.sum((K.expand_dims(y_true,2) - mu)**2, axis=1)/(2*(sigma)**2)
    
    log_gauss = log_sum_exp(exponent, axis=1)
    res = - K.mean(log_gauss)
    return res


global_loss_list['mean_log_Gaussian_like']=mean_log_Gaussian_like


def mean_log_LaPlace_like(y_true, parameters):
    """Mean Log Laplace Likelihood distribution
    Note: The 'c' variable is obtained as global variable
    """
    #Note: The output size will be (c + 2) * m = 6
    c = 1 #The number of outputs we want to predict
    m = 2 #The number of distributions we want to use in the mixture
    components = K.reshape(parameters,[-1, c + 2, m])
    mu = components[:, :c, :]
    sigma = components[:, c, :]
    alpha = components[:, c + 1, :]
    alpha = K.softmax(K.clip(alpha,1e-2,1.))
    
    exponent = K.log(alpha) - float(c) * K.log(2 * sigma) \
    - K.sum(K.abs(K.expand_dims(y_true,2) - mu), axis=1)/(sigma)
    
    log_gauss = log_sum_exp(exponent, axis=1)
    res = - K.mean(log_gauss)
    return res

global_loss_list['mean_log_LaPlace_like']=mean_log_LaPlace_like

