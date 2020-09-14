
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math

class RBF():

    def __init__(self,dim,seed=42):
        self.dim=dim
        self.seed=seed
        self.weights=None

    def generateData(self,x):
        sinus=np.sin(2*x)
        square=signal.square(2*x)
        return sinus,square

    def initWeights(self,sigma=0.1):
        weights = np.random.RandomState(seed=self.seed).randn(1, self.dim)*sigma
        self.weights=np.transpose(weights)
        return self.weights
        

    def deltaRule(self,trainingData,targetData):
        print("Sequantial delta rule")

 
    def transferFunction(self,x,mu,sigma):
        PHI = np.zeros((x.shape[0], mu.shape[0]))
        for i in range(x.shape[0]):
            phi = np.exp((-(x[i]-mu)**2)/(2*sigma**2))
            PHI[i,:] = phi
        return PHI

    def activationFunction(self,weights,phi):
        function= phi @ weights
        return function


    def leastSquares(self, PHI, function):
        # rest=trainingData%batchSize
        # batches = [trainingData[i*batchSize:(i+1)*batchSize] for i in range(int(trainingData.shape[0]/batchSize))]
        # batches.append(trainingData[-rest,-1])
        self.weights = np.linalg.lstsq(PHI, function)
        return self.weights[0],self.weights[1]
        
    def train(self,xtrain,ytrain,weights,mu, sigma):
        phi=self.transferFunction(xtrain,mu,sigma)
        function=self.activationFunction(weights,phi)
        weights, error=self.leastSquares(phi, function)
        return weights, error

    def evaluation(self,xtest,weights,mu, sigma):
        phi=self.transferFunction(xtest,mu,sigma)
        return self.activationFunction(weights,phi)

    def classify(self):

        pass

def main():
    ## generate data and define inputs
    mu=np.array([3,6,2,5,3,7,6])
    sigma=0.1
    trainingData=np.arange(0,2*math.pi,0.1)
    testData=np.arange(0.05,2*math.pi,0.1)
    ## init rbf class
    dim=mu.shape[0]
    rbf=RBF(dim)
    ## 

    sinus,square    = rbf.generateData(trainingData)
    weights         = rbf.initWeights()
    weights, error  = rbf.train(trainingData, sinus, weights, mu, sigma)
    ytest           = rbf.evaluation(testData, weights, mu, sigma)

    plt.plot(testData, ytest)
    plt.show()
if __name__ == "__main__":
    main()
