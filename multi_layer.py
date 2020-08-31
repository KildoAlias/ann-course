import matplotlib.pyplot as plt
import numpy as np
import math
from generateData import generateClassData
from progressBar import progressBar
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

    def initWeights(self, dim=2, sigma=0.1):
        """
        dim: The dimension of the input layer\n
        sigma: Default value 0.1.
        """
        # Init weights for hidden layers
        for layer in self.layers:
            if self.bias:
                dim += 1
            self.weights.append(np.random.RandomState(
                seed=self.seed*10).randn(layer, dim)*sigma)
            dim = layer

    def transferFunction(self, x):
        return 2 / (1 + np.exp(-x)) - 1

    def activationFunction(self, data):
        data[data > 0] = 1
        data[data < 0] = -1
        return data

    def forwardpass(self, data, layer):
        """
        Description:\n
            forwardpass function(recursive function), first part of the classification\n
        Input:\n
            data: the intput data for current layer
            layer: current layer (number)
        output:\n
            out: output from the neural network
        """
        if layer == self.numberOfLayers:
            return data
        else:
            hin = self.weights[layer] @ np.concatenate(
                (self.transferFunction(data), np.ones((1, np.shape(data)[1]))), axis=0)
            hout = self.transferFunction(hin)
            return self.forwardpass(data=hout, layer=layer+1)

    def classify(self, data):
        data = self.forwardpass(data=data, layer=0)
        return self.activationFunction(data)

    def eval(self, data, targets, verbose=False):
        classified_data = self.classify(data)
        accuracy = np.count_nonzero(
            classified_data == targets)/np.shape(targets)[1]*100
        print("Accuracy: ", accuracy)
        if verbose:
            plt.scatter(data[0, np.where(classified_data==targets)], data[1, np.where(classified_data==targets)], c="green")
            plt.scatter(data[0, np.where(classified_data!=targets)], data[1, np.where(classified_data!=targets)], c="red")
            plt.show()

    def loss_val(self, data, target):
        loss = 1 / (2*np.shape(target)[1]) * np.sum( np.power(self.forwardpass(data=data, layer=0) - target, 2))
        return loss

    def train(self, data, targets, epochs, eta=0.001, alpha=0.9):
        def forwardpass(data, layer, out_vec=[]):
            """
            Description:\n
                Forwardpass function (recursive function)\n
            Input:\n
                data: the intput data for current layer
                layer: current layer (number)
                out_vec: the output vector with corresponding output
            """
            if layer == self.numberOfLayers:
                out_vec.append(data)
                return out_vec[1:]
            else:
                out_vec.append(np.concatenate(
                    (data, np.ones((1, np.shape(data)[1])))))
                hin = self.weights[layer] @ np.concatenate(
                    (self.transferFunction(data), np.ones((1, np.shape(data)[1]))), axis=0)
                hout = self.transferFunction(hin)
                return forwardpass(data=hout, layer=layer+1, out_vec=out_vec)

        def backprop(out_vec, targets):
            """
            Description:\n
            Backprop function\n
            Input:\n
                out_vec: the output vector for each layer\n
                targets: target label\n

            Output:\n
                delta_h: the delta for the hidden layer\n
                delta_o: the delta for the output layer\n
            """
            delta_o = (out_vec[-1] - targets) * \
                ((1 + out_vec[-1]) * (1 - out_vec[-1])) * 0.5
            delta_h = (np.transpose(
                self.weights[-1]) @ delta_o) * ((1 + out_vec[0]) * (1 - out_vec[0])) * 0.5
            delta_h = delta_h[0:self.layers[0], :]
            return delta_h, delta_o

        # Inital delta weights.
        dw = np.ones(np.shape(self.weights[0]))
        dv = np.ones(np.shape(self.weights[1]))
        
        progBar = progressBar(epochs,10)
        loss_vec = []
        epoch_vec = []
        # training for all the epochs.
        for epoch in range(epochs):
            progBar.Progress(epoch)
            # Forwarding
            out_vec = forwardpass(data=data, layer=0)

            # Back propogating
            delta_hidden, delta_output = backprop(out_vec, targets)

            # Weights update
            pat = np.concatenate((data, np.ones((1, np.shape(data)[1]))))
            dw = (dw * alpha) - (delta_hidden @
                                 np.transpose(pat)) * (1 - alpha)
            dv = (dv * alpha) - (delta_output @
                                 np.transpose(out_vec[0])) * (1 - alpha)

            self.weights[0] = self.weights[0] + dw*eta
            self.weights[1] = self.weights[1] + dv*eta

            loss_vec.append(self.loss_val(data, target=targets))
            epoch_vec.append(epoch)
        return epoch_vec, loss_vec



def main():
    n = 100
    bias = True
    data, targets = generateClassData(n, proc_A=1, proc_B=1, verbose=False, seed=100,linear=False)

    NN = neuralNetwork(bias=bias, layers=[100, 1])
    NN.initWeights()
    print(NN)
    epoch_vec, loss_vec = NN.train(data, targets=targets, epochs=10000, eta=0.001)

    plt.plot(epoch_vec, loss_vec)
    plt.show()
    NN.initWeights()
    NN.eval(data, targets, verbose=True)


if __name__ == "__main__":
    main()