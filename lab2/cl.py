from rbf import RBF
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math


#! Cl Class ONLY FOR 1-D DATA
class CL():
    def __init__(self,nUnits,data,width,steps,learningRate,show=False, info=True, winners=1):
        self.nUnits=nUnits
        self.weights=None
        self.data=data
        self.nDataPoints=len(data)
        self.trainingData=None
        self.width=width
        self.steps=steps
        self.learningRate=learningRate
        self.show=show
        self.step=None
        self.winners=np.zeros(winners,dtype=int)
        if self.show:
            self.y=np.zeros(nUnits)
            plt.show()
        if info:
            print(self)
        

    def __str__(self):
        return 'CL class: \n Units: {} \n Datapoints: {} \n Dimensions: {} \n Winners: {}'.format(self.nUnits, self.nDataPoints, self.data.shape, self.winners.shape[0])
    
    def initWeights(self):
        weights = np.random.uniform(0,1, self.nUnits)* np.amax(self.data)
        self.weights=np.transpose(weights)

    def trainingVector(self):
        #! in 1-D it is only a datapoint
        index=np.random.randint(0,self.nDataPoints)
        self.trainingData=self.data[index]

    def selection(self):

        distance=[np.abs(self.trainingData-weight) for weight in self.weights]

        for i in range(self.winners.shape[0]):
            winnerValue =np.amin(distance)    
            self.winnerIndex = np.where(distance == winnerValue)[0][0]
            self.winners[i]=int(self.winnerIndex)
            distance.pop(self.winnerIndex)

    def update(self):

        for winner in self.winners:
            self.weights[winner]=self.weights[winner]+self.learningRate*(self.trainingData-self.weights[winner])


    

    def train(self):
        self.initWeights()
        for self.step in range(self.steps):
            self.trainingVector()
            self.selection()
            self.update()
            if self.show:
                self.plot()

    
    def plot(self):
        plt.clf()
        plt.title(self.step)
        plt.scatter(self.weights,self.y,c="b")
        plt.scatter(self.trainingData,0,c="r")
        plt.legend(["Weights" , "Sampled data"])
        plt.pause(0.1)
        plt.axis(xmin=0,xmax=10)







