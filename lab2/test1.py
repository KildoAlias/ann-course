from rbf import RBF
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math

# Add zero-mean gaussian noize with variance 0.1



def sinus_LS(x_test, x_train, mu, sigma):
    dim=mu.shape[0]
    rbf = RBF(dim)
    sinus, _   = rbf.generateData(x_train)
    sinus_test, _ = rbf.generateData(x_test)

    ## Init and train.
    weights         = rbf.initWeights()
    weights, error  = rbf.train_LS(x_train, sinus, weights, mu, sigma)
    
    ## Evaluation 
    y_test = rbf.evaluation_LS(x_test, weights, mu, sigma)
    residual_error = np.sum(abs(y_test - sinus_test))/y_test.shape[0]
    print('Residual error: ', residual_error)
    return residual_error

def square_LS(x_test, x_train, mu, sigma):
    pass

def sinus_delta(x_test, x_train, mu, sigma):
    pass

def square_delta(x_test, x_train, mu, sigma):
    pass

def main():
    # GENERATES DATASET (TRAIN & TEST)
    x_train = np.arange(0,2*math.pi,0.1)
    x_test = np.arange(0.05,2*math.pi,0.1)

    # KERNEL PARAMS 
    mu = np.arange(0,2*math.pi,0.3)
    sigma = 0.1

    # SINUS LEAST SQUARE
    averages = 10

    

    sinus_LS(x_test, x_train, mu, sigma)

    # SQUARE LEAST SQUARE

    # SINUS DELTA-RULE

    # SQUARE DELTA-RULE

if __name__ == '__main__':
    main()

