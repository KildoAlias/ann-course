import matplotlib.pyplot as plt
import numpy as np
import math


class neuralNetwork():
    def __init__(self, layers, bias=True, seed=42):
        """
        hiddenLayers: Number of hiddenLayers [neurons for lvl1, ... etc]\n
        bias: True/False\n
        seed: seed number\n
        """
        self.layers = layers
        self.numberOfLayers = len(layers)
        self.weights = []
        self.bias = bias
        self.seed = seed

    def __str__(self):
        """
        What information that should be shown about the NN is stated here.
        Structure: the structure of the NN.
        Bias: True/False
        """
        structure = "structure : {} \n".format(
            [np.shape(np.transpose(w)) for w in self.weights])
        layers = "Layers (neurons): {} \n".format(
            self.layers)

        bias = "Bias: {} \n".format(self.bias)
        return structure + layers + bias

    def initWeights(self, dim=2, sigma=1):
        """
        dim: The dimension of the input layer\n
        sigma: Default value 1.
        """
        # Init weights for hidden layers
        for layer in self.layers:
            if self.bias:
                dim += 1
            self.weights.append(np.random.RandomState(
                seed=self.seed).randn(layer, dim)*sigma)
            dim = layer

    def transferFunction(self, x):
        return 2 / (1 + np.exp(-x)) - 1

    def activationFunction(self):
        return

    def classify(self):
        return

    def eval(self):
        return

    def train(self, data, epochs):
        def forwardpass(patterns, layer):
            if layer == self.numberOfLayers:
                return patterns
            else:
                hin = self.weights[layer] @ np.concatenate(
                    (self.transferFunction(patterns), np.ones((1, np.shape(patterns)[1]))), axis=0)
                hout = self.transferFunction(hin)
                return forwardpass(hout, layer+1)

        def backprop():
            return

        for epoch in range(epochs):
            out = forwardpass(data, 0)
            print(out)


def generateClassData(nA, nB, mA, mB, sigmaA, sigmaB, bias, seed=42):

    classA_size = mA.size
    classA = np.random.RandomState(seed=seed).randn(
        classA_size, nA)*sigmaA + np.transpose(np.array([mA]*nA))

    classB_size = mB.size
    classB = np.random.RandomState(seed=seed).randn(
        classB_size, nB)*sigmaB + np.transpose(np.array([mB]*nB))

    classData = np.concatenate((classA, classB), axis=1)

    T = np.concatenate((np.ones((1, nA)), np.ones((1, nB))*(-1)), axis=1)

    shuffler = np.random.RandomState(seed=seed).permutation(nA+nB)
    classData = classData[:, shuffler]
    T = T[:, shuffler]

    return classData, T


mA = np.array([1, 0.5])
mB = np.array([-1, 0])
nA = 100
nB = 100
sigmaA = 0.4
sigmaB = 0.4
bias = True

data, target = generateClassData(nA, nB, mA, mB, sigmaA, sigmaB, bias=bias)
NN = neuralNetwork(bias=bias, layers=[2, 1])
NN.initWeights()
print(NN)
NN.train(data, 10)
NN.initWeights()
