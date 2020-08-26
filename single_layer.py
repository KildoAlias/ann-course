## Parameters: 
#? epochs (20 suitable)
#? input size
#? output size
#? number of training patterns
#? step length (learning  rate eta, small value, for example 0.001)
#! a common mistake when implementing this is to accidentally orient the matrixes wrongly so that columns and rows are interchanged
#! Have initial  values  assigned to weights (small  random numbers drawn from the normal distribution with zero mean) (Note  that  the  matrix  must  have matching dimensions)

import numpy as np
import matplotlib.pyplot as plt


def generate_data(n,mA,mB,sigmaA,sigmaB):

    classA_size=mA.size
    classA=np.random.randn(classA_size,n)*sigmaA + np.transpose(np.array([mA]*n))

    classB_size=mB.size
    classB=np.random.randn(classB_size,n)*sigmaB + np.transpose(np.array([mB]*n))

    classAB = np.concatenate((classA,classB),axis=1)
    T = np.concatenate((np.ones((1,200)), np.ones((1,200))*(-1)),axis=1)
    data = np.concatenate((classAB, np.ones((1,400)), T), axis=0)

    return classA, classB, data

def plot_data(classA,classB):
    plt.scatter(classB[0,:],classB[1,:],c="red")
    plt.scatter(classA[0,:],classA[1,:], c="blue")
    plt.show()
    return


def main():
    ####### Parameters for generate_data
    mA=np.array([1,0.5])
    mB=np.array([-1,0])
    n=200
    sigmaA=0.5
    sigmaB=0.5
    #######
    classA, classB,data =generate_data(n,mA,mB, sigmaA, sigmaB)
    plot_data(classA,classB)



def generate_weights():


main()





