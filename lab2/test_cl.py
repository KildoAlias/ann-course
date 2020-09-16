from rbf import RBF
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math
from cl import CL


def generateData(noisy,sigma):
    data = np.arange(0,2*math.pi,0.2)
    if noisy:
        data=data+np.random.randn(data.shape[0])*sigma
    return data




def main():
    data=generateData(noisy=False,sigma=0.1)
    cl=CL(64,data,0.1,1000,0.1,show=True,winners=4)
    cl.train()



    

if __name__ == "__main__":
    main()

    