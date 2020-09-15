from rbf import RBF
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math
from cl import CL




def main():
    x = np.arange(0,2*math.pi,0.2)
    y=np.zeros(16)
    cl=CL(16,x,0.1,1000,1)
    cl.train()



    

if __name__ == "__main__":
    main()

    