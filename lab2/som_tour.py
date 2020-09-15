import numpy as np
import random
import matplotlib.pyplot as plt


class SOM():

    def __init__(self, nNodes, inputDim, nClass, eta=1):  # eta scrambled input = 1.1
        self.nNodes = nNodes
        self.weights = None
        self.eta = eta
        self.inputDim = inputDim
        self.nClass = nClass

    def initWeights(self):
        self.weights = np.zeros((self.nNodes, self.inputDim))
        for i in range(self.nNodes):
            self.weights[i][:] = np.random.uniform(0, 1, self.inputDim)
        return self.weights

    def euclidianDist(self, pattern):
        dBest = 100000000
        # print(pattern
        iBest = 0
        for i in range(self.weights.shape[0]):
            # d = np.transpose(
            #     pattern-self.weights[i][:])@(pattern-self.weights[i][:])
            d = np.linalg.norm(pattern-self.weights[i, :])
            if d < dBest:
                dBest = d
                iBest = i
        return iBest

    def neighbourhood(self, index, epoch, epochs):
        if epoch/epochs <= 0.1:
            dist = 2
        elif epoch/epochs <= 0.6:
            dist = 1
        else:
            dist = 0

        neighbours = np.linspace(
            index-dist, index+dist, 2*dist+1)
        # print(" neigh befoer = ", neighbours)
        neighbours = np.where(neighbours < 0, neighbours + 10, neighbours)
        neighbours = np.where(neighbours > 9, neighbours - 10, neighbours)
        # print(" neigh = ", neighbours)
        # print()
        return neighbours

    def weightsUpdate(self, pattern, neighbours):
        for i in neighbours:
            # print(i)
            # print('weight=', self.weights[int(i)][:])
            # print('update=', self.eta*(pattern-self.weights[int(i)][:]))

            self.weights[int(i)][:] = self.weights[int(i)][:] + \
                self.eta*(pattern-self.weights[int(i)][:])
            # print('new weight=', self.weights[int(i)][:])
        return self.weights


def main():

    cityData = np.array([[0.4000, 0.4439], [0.2439, 0.1463], [0.1707, 0.2293],  [0.2293, 0.7610], [0.5171, 0.9414], [
                        0.8732, 0.6536], [0.6878, 0.5219], [0.8488, 0.3609], [0.6683, 0.2536], [0.6195, 0.2634]])

    plt.figure('map')
    index = 0
    for data in cityData:
        plt.scatter(data[0], data[1], c='red')
        plt.text(data[0], data[1], str(index))
        index += 1

    ####### init som and weights ###########
    som = SOM(nNodes=10, inputDim=2, nClass=10)
    weights = som.initWeights()
    epochs = 40
    ######################################

    ######## Training ###################
    for epoch in range(epochs):
        shuffler = np.random.permutation(10)
        datapoint = np.arange(0, 10, 1)
        datapoint = datapoint[shuffler]
        for i in datapoint:
            iBest = som.euclidianDist(cityData[i][:])
            # print(iBest)
            neighbours = som.neighbourhood(iBest, epoch, epochs)
            # print(neighbours)
            weights = som.weightsUpdate(cityData[i][:], neighbours)
        som.eta *= 0.95
    # ######################################

    # ########## Testing ##################
    winnerIndexes = []
    for i in range(10):
        iBest = som.euclidianDist(cityData[i][:])
        # print(iBest)
        winnerIndexes.append(iBest)
    # ######################################

    winnerIndexes = sorted(winnerIndexes)

    print(winnerIndexes)
    X = [cityData[i, 0] for i in winnerIndexes]
    Y = [cityData[i, 1] for i in winnerIndexes]
    plt.plot(X, Y)

    # cityData = [x for _, x in sorted(zip(winnerIndexes, cityData))]
    # print(cityData)
    index = 0
    for data in weights:
        plt.scatter(data[0], data[1], c='green')
        plt.text(data[0], data[1], str(index))
        index += 1

    plt.show()

    # print(winnerIndexes)


if __name__ == "__main__":
    main()
