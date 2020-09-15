from rbf import RBF
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math


#! Cl Class ONLY FOR 1-D DATA
class CL():
    def __init__(self,nUnits,data,width,steps,learningRate):
        self.nUnits=nUnits
        self.weights=None
        self.data=data
        self.nDataPoints=len(data)
        self.trainingData=None
        self.width=width
        self.steps=steps
        self.learningRate=learningRate
        

    def __str__(self):
        return 'CL class with {} units and {} datapoints'.format(self.nUnits, self.nDataPoints)
    
    def initWeights(self):
        weights = np.random.uniform(0,1, self.nUnits)* np.amax(self.data)
        self.weights=np.transpose(weights)
        print("Weights initialized with variance ", self.width)

    def trainingVector(self):
        #! in 1-D it is only a datapoint
        index=np.random.randint(0,self.nDataPoints)
        self.trainingData=self.data[index]

    def selection(self):
        distance=[self.trainingData-weight for weight in self.weights]
        self.winnerValue =np.amin(np.absolute(distance))    
        self.winnerIndex = np.where(distance == self.winnerValue)


    def update(self):
        self.weights[self.winnerIndex]=self.winnerValue+self.learningRate*(self.trainingData-self.winnerValue)

    def train(self):
        y=np.zeros(16)
        self.__str__()
        self.initWeights()
        for step in range(self.steps):
            self.trainingVector()
            self.selection()
            plt.clf()
            plt.scatter(self.weights,y,c="b")
            plt.title(step)
            plt.pause(0.001)
            self.update()
        plt.show()
        plt.axis(xmin=0,xmax=10)
        print("Training completed after", self.steps, "steps")
        








