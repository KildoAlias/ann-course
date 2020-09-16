from rbf import RBF
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math


#! Cl Class ONLY FOR 1-D DATA
class CL():
    def __init__(self,nUnits,data,width,steps,learningRate,show=False, info=True, nbSize=0):
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
        self.nbSize=nbSize
        self.neighbours=np.zeros(nbSize)
        if self.show:
            self.y=np.zeros(nUnits)
            plt.show()
        if info:
            print(self)
        

    def __str__(self):
        return 'CL class: \n Units: {} \n Datapoints: {} \n Dimensions: {} \n Neighbours: {}'.format(self.nUnits, self.nDataPoints, self.data.shape, self.nbSize)
    
    def initWeights(self):
        weights = np.random.uniform(0,1, self.nUnits)* np.amax(self.data)
        self.weights=np.transpose(weights)

    def trainingVector(self):
        #! in 1-D it is only a datapoint
        index=np.random.randint(0,self.nDataPoints)
        self.trainingData=self.data[index]

    def selection(self):
        distance=[self.trainingData-weight for weight in self.weights]
        # distance=np.absolute(distance)
        sortedDistance=sorted(distance[:])
        self.winnerValue =sortedDistance[0]    
        self.winnerIndex = np.where(distance == self.winnerValue)[0][0]
        if self.nbSize !=0:
            neighbours=sortedDistance[1:self.nbSize+1]
            for i,  neighbour in enumerate(neighbours):
                self.neighbours[i]=np.where(distance == neighbour)[0][0]

    def update(self):
        self.weights[self.winnerIndex]=self.winnerValue+self.learningRate*(self.trainingData-self.winnerValue)
        if self.nbSize !=0:
            for neighbour in self.neighbours:
                neighbour=int(neighbour)
                self.weights[neighbour]=self.weights[neighbour]+self.learningRate*0.5*(self.trainingData-self.weights[neighbour])
    

    

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
        plt.pause(0.0001)
        plt.axis(xmin=0,xmax=10)







