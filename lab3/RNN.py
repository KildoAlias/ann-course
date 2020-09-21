import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits import mplot3d
import itertools
import time
# from alive_progress import alive_bar


class RNN():
    def __init__(self, size=8, diagonal=True, sequential=False, random=True):
        self.size = size
        self.diagonal = diagonal
        self.weights = np.zeros((size,size))
        self.sequential=sequential
        self.random = random

    
    def init_weights(self, patterns):
        if self.sequential and self.random:
            random_vec = np.random.permutation(np.linspace(0, self.size - 1, self.size))
            for pattern in patterns:
                for i in random_vec:
                    i = int(i)
                    for j in random_vec:  
                        j = int(j)
                        self.weights[i][j] += pattern[0][i]*pattern[0][j]
        
        elif self.sequential and not self.random:
            for pattern in patterns:
                for i in range(self.size):
                    for j in range(self.size):  
                        self.weights[i][j] += pattern[0][i]*pattern[0][j]
        elif self.diagonal:
            for x in patterns:
                self.weights += (np.transpose(x) @ x)
            self.weights /= self.size
        else:
            for x in patterns:
                self.weights += (np.transpose(x) @ x)

            self.weights /= self.size
            for i in range(self.weights.shape[0]):                                  # Removes diagonal elements
                self.weights[i,i] = 0 

    def update(self, x):
        if self.sequential:
            x_new=np.zeros((x.shape[0],x.shape[1]))
            for i in range(self.size):
                for j in range(self.size):
                    x_new[0][i] += self.weights[i][j]*x[0][j]
                x_new = np.sign(x_new)
            x_new = np.transpose(x_new)
        else:    
            x_new = np.sign(self.weights @ np.transpose(x))
        return np.transpose(x_new)

    def train(self, x):
        x_old = np.zeros((1,self.size))                                           # HÅRD SOM FAN
        x_new = x  
        iteration = 0
        while not (x_new == x_old).all():
            iteration += 1
            x_old = x_new
            x_new = self.update(x_old)
        print("Iteration: {}".format(iteration))
        # print("Results: ", x_new)
        return x_new

    def plot_weights(self): # JÄVLIGT HÅRD
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        X, Y = np.meshgrid(np.arange(0,8,1), np.arange(0,8,1))
        ax.plot_surface(X, Y, self.weights, rstride=1, cstride=1,
                cmap='jet', edgecolor='none')

        ax.set_title('finaplot');







def main():
    # Original patterns
    x1 = np.array([[-1, -1, 1, -1, 1, -1, -1 ,1]])
    x2 = np.array([[-1, -1, -1, -1, -1, 1, -1 ,-1]])
    x3 = np.array([[-1, 1, 1, -1, -1, 1, -1 ,1]])
    print(x1.shape)
    # Distorted patterns
    x1d = np.array([[1, -1, 1, -1, 1, -1, -1 ,1]])
    x2d = np.array([[1, 1, -1, -1, -1, 1, -1 ,-1]])
    x3d = np.array([[1, 1, 1, -1, 1, 1, -1 ,1]])
    
    patterns = [x1, x2, x3]
    patterns_distorted = [x1d, x2d, x3d]
    network = RNN()
    network.init_weights(patterns)
    print(patterns)

    for xd, x in zip(patterns_distorted, patterns):
        x_output = network.train(xd)
        print('Number of correct: {}/{} '.format(np.count_nonzero(x_output==x), x.shape[1]))


    print('\nMore than half distorted: ')
    x4d = np.array([[1, -1, 1, -1, 1, 1, -1 ,-1]])
    x_output = network.train(x4d)
    print('Number of correct: {}/{} '.format(np.count_nonzero(x_output==x1), x1.shape[1]))
    print('Number of correct: {}/{} '.format(np.count_nonzero(x_output==x2), x2.shape[1]))
    print('Number of correct: {}/{} '.format(np.count_nonzero(x_output==x3), x3.shape[1]))
    print('Output: ', x_output)


    # Taking out all possible attractors
    lst = list(itertools.product([-1, 1], repeat=8))
    attractors = []
    for xd in lst:
        xd = np.array([xd])
        x_output = network.train(xd)
        x_output = np.ndarray.tolist(x_output)
        if not np.all(x_output in attractors):
            attractors.append(x_output)
    
    print('\nNumber of unique attractors: ',len(attractors))
















if __name__ == '__main__':
    main()
