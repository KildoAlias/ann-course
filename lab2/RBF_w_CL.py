from cl import CL
from test_cl import readData
from rbf_cl import RBF


def main():

    # GENERATE DATASET
    x_train, y_train =readData()
    cl=CL(100,x_train,0.1,1000,0.4,show=False,winners=3)
    cl.train()

    # PARAMETERS
    mu = cl.weights
    sigma = 0.1

    # INIT RBF
    rbf = RBF(2)
    weights = rbf.initWeights()

    # TRAIN RBF
    rbf.train_DELTA(x_train=x_train,
                    y_train=y_train,
                    weights=weights,
                    mu=mu,
                    sigma=sigma)


if __name__ == "__main__":
    main()



