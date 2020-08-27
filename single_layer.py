

import numpy as np
import matplotlib.pyplot as plt


class perceptron():

    def __init__(self, layers, bias=True, batch=True, learningRate=0.001, seed=42):
        self.layers = layers
        self.bias = bias
        self.batch = batch
        self.classData = None
        self.T = None
        self.classA = None
        self.classB = None
        self.W = None
        self.learningRate = learningRate
        self.seed = seed
        return

    def generateClassData(self, nA, nB, mA, mB, sigmaA, sigmaB):

        classA_size = mA.size
        self.classA = np.random.RandomState(seed=self.seed).randn(
            classA_size, nA)*sigmaA + np.transpose(np.array([mA]*nA))

        classB_size = mB.size
        self.classB = np.random.RandomState(seed=self.seed).randn(
            classB_size, nB)*sigmaB + np.transpose(np.array([mB]*nB))

        classAB = np.concatenate((self.classA, self.classB), axis=1)
        # X in math
        if self.bias:
            classData = np.concatenate((classAB, np.ones((1, nA+nB))), axis=0)
        else:
            classData = classAB

        T = np.concatenate((np.ones((1, nA)), np.ones((1, nB))*(-1)), axis=1)

        shuffler = np.random.RandomState(seed=self.seed).permutation(nA+nB)
        self.classData = classData[:, shuffler]
        self.T = T[:, shuffler]

        return

    def plotData(self, epoch=None):

        plt.scatter(self.classA[0, :], self.classA[1, :], c="red")
        plt.scatter(self.classB[0, :], self.classB[1, :], c="blue")
        y1 = -self.W[0, 2]/self.W[0, 1] - self.W[0, 0]/self.W[0, 1]*(-2)
        y2 = -self.W[0, 2]/self.W[0, 1] - self.W[0, 0]/self.W[0, 1]*(2)
        plt.plot([-2, 2], [y1, y2])
        plt.axis([-2, 2, -2, 2])
        plt.title("Epoch: {}, Batch: {}".format(epoch, self.batch))
        plt.draw()
        plt.pause(0.1)
        return

    def initWeights(self, dim=2, sigma=0.1):
        if self.bias:
            dim += 1
        self.W = np.random.RandomState(seed=self.seed).randn(1, dim)*sigma

        return

    def deltaRule(self):
        if self.batch:
            deltaW = -self.learningRate * \
                np.dot((np.dot(self.W, self.classData) - self.T),
                       np.transpose(self.classData))
            self.W += deltaW
        else:
            deltaW = np.zeros(np.shape(self.W))
            for i in range(np.shape(self.classData)[1]):
                deltaW += -self.learningRate * \
                    (np.dot(
                        self.W, self.classData[:, i]) - self.T[:, i])*np.transpose(self.classData[:, i])
            self.W += deltaW

    def train(self, epochs=20, verbose=True):
        for i in range(epochs):
            self.deltaRule()
            if verbose:
                self.plotData(i)
        return

        # Parameters:
        # ? epochs (20 suitable)
        # ? input size
        # ? output size
        # ? number of training patterns
        # ? step length (learning  rate eta, small value, for example 0.001)
        #! a common mistake when implementing this is to accidentally orient the matrixes wrongly so that columns and rows are interchanged
        #! Have initial  values  assigned to weights (small  random numbers drawn from the normal distribution with zero mean) (Note  that  the  matrix  must  have matching dimensions)


def main():
    # Parameters for generate_data
    mA = np.array([1, 0.5])
    mB = np.array([-1, 0])
    nA = 200
    nB = 200
    sigmaA = 0.5
    sigmaB = 0.5
    #######
    single_layer = perceptron(1, batch=True)
    single_layer.generateClassData(nA, nB, mA, mB, sigmaA, sigmaB)
    single_layer.initWeights()
    # single_layer.plotData()
    single_layer.train()


if __name__ == "__main__":
    main()
