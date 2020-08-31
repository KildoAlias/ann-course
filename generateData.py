import numpy as np
import matplotlib.pyplot as plt
def generateClassData(n, proc_A, proc_B, seed=42, linear=False, verbose=True):
    if linear:
        mA = np.array([1, 0.5])
        mB = np.array([-1, 0])
        sigmaA = 0.5
        sigmaB = 0.5

        # Take out the size of the classes
        classA_size = mA.size
        classB_size = mB.size

        # Create datapoints for A and B
        classA = np.random.RandomState(seed=seed).randn(
            classA_size, n)*sigmaA + np.transpose(np.array([mA]*n))
        classB = np.random.RandomState(seed=seed).randn(
            classB_size, n)*sigmaB + np.transpose(np.array([mB]*n))

        # Merges the datapoints from class A and class B and adding a target vector
        classData = np.concatenate((classA, classB), axis=1)
        T = np.concatenate((np.ones((1, n)), np.ones((1, n))*(-1)), axis=1)

        # Shuffler the data and the targets
        shuffler = np.random.RandomState(seed=seed).permutation(n+n)
        classData = classData[:, shuffler]
        T = T[:, shuffler]

        if verbose:
            plt.scatter(classA[0,:], classA[1,:], c="red")
            plt.scatter(classB[0,:], classB[1,:], c="blue")
            plt.show()

        return classData, T
    elif not linear:
        mA = np.array([1.0, 0.3])
        mB = np.array([0.0, -0.1])
        sigmaA = 0.05
        sigmaB = 0.05

        # Creates class A data
        classA = np.random.RandomState(seed=seed).randn(1, round(0.5*n)) * sigmaA - mA[0]
        classA = np.concatenate((classA, np.random.RandomState(seed=seed).randn(1, round(0.5*n)) * sigmaA + mA[0]), axis=1)
        classA = np.concatenate((classA, np.random.RandomState(seed=seed+1).randn(1, n) * sigmaA + mA[1]), axis=0)
    
        # Creates class B data
        classB = np.random.RandomState(seed=seed).randn(1, n) * sigmaB + mB[0]
        classB = np.concatenate((classB, np.random.RandomState(seed=seed+1).randn(1, n) * sigmaB + mB[1]), axis=0)

        # Creates target data
        T_A = np.ones((1, n))
        T_B = np.ones((1, n))*(-1)

        # Shuffles the data sets
        shuffler = np.random.RandomState(seed=seed).permutation(n)
        classA = classA[:, shuffler]
        classB = classB[:, shuffler]
        T_A = T_A[:, shuffler]
        T_B = T_B[:, shuffler]

        # Take out the wanted procentage from each dataset
        classA = classA[:round(n*proc_A),:]
        classB = classB[:round(n*proc_B),:]
        T_A = T_A[:round(n*proc_A),:]
        T_B = T_B[:round(n*proc_A),:]
        
        # concatenates the datasets and the targets
        data = np.concatenate((classA, classB), axis=1)
        T = np.concatenate((T_A, T_B), axis=1)

        # Shuffles the dataset and the target
        shuffler = np.random.RandomState(seed=seed).permutation(round(n*proc_A) + round(n*proc_B))
        data = data[:, shuffler]
        T = T[:, shuffler]
        
        if verbose:
            plt.scatter(classA[0,:], classA[1,:], c="red")
            plt.scatter(classB[0,:], classB[1,:], c="blue")
            plt.show()

        return data, T
