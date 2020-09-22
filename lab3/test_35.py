from RNN import RNN
import numpy as np
import math
from mpl_toolkits import mplot3d
import itertools
import matplotlib.pyplot as plt
from test_33 import loadData

def distort(pattern,noise):
    patternD=pattern.copy()
    patternD=np.squeeze(patternD)
    size=patternD.shape[0]
    noise=int(np.floor(noise/100*size))
    index=np.random.randint(0,size,noise)
    distorted=np.where(patternD[index]==1, -1, 1)
    patternD[index]=distorted
    patternD=np.reshape(patternD,[1,patternD.shape[0]])
    return patternD

def randomPatterns(nPatterns, size):
    rndPatterns=np.random.choice([-1,1],[nPatterns,size])
    return rndPatterns




def main():

    # patterns = loadData('pict.dat')     # Pattern 1-11
    patterns=randomPatterns(nPatterns=300, size=1024)
    print(patterns.shape)
    patterns_1_3 = [patterns[index,:].reshape(1,1024) for index in range(300) ]
    # patterns_4_11 = [patterns[3+index,:].reshape(1,1024) for index in range(8) ]
    

    network = RNN(size=1024, sequential=False, random=False)
    network.init_weights(patterns_1_3)
    noises=np.arange(0,101,5)
    averages=2
    for i, pattern in enumerate(patterns_1_3):
        OGpattern=pattern.copy()
        print("Pattern P:",i+1)
        nCorrect=np.zeros((noises.shape[0],1))
        for k, noise in enumerate(noises):
            for j in range(averages):
                patternD=distort(OGpattern,noise)
                x_output = network.train(patternD)
                nCorrect[k][0] += ((np.count_nonzero(x_output==OGpattern))/patternD.shape[1])*100

        nCorrect=nCorrect/averages
        # plt.imshow(x_output.reshape(32,32), cmap='gray')
        # plt.show()
        plt.plot(noises,nCorrect, label=("Pattern " + str(i+1)))
    plt.axis(xmin=0,xmax=100, ymin=0, ymax=100)
    plt.xlabel("Noise [%]")
    plt.ylabel("Accuracy [%]")
    plt.title("Capacity")
    # plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

