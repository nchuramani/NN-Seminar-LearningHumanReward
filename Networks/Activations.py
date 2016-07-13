# -*- coding: utf-8 -*-
import theano
import theano.tensor as T

from Utils import DataUtil


def applyActivationFunction(activationFunction, input):
    
    if activationFunction == DataUtil.ACTIVATION_FUNCTION["Tanh"]:        
        return tanh(input)
    elif activationFunction == DataUtil.ACTIVATION_FUNCTION["ReLU"]:
        return reLU(input)


def tanh(x): 
    return (T.tanh(x))

def reLU(x):    
    #y = T.maximum(x,0)
    
    return(T.maximum(T.cast(0., theano.config.floatX), x))