from cl import CL
from test_cl import readData
from rbf_cl import RBF
import matplotlib.pyplot as plt


def main():

    # GENERATE DATASET
    x_train, y_train =readData('ballist.dat')
    x_test, y_true =readData('balltest.dat')
    cl=CL(10,x_train,0.1,1000,0.4,show=False,winners=3)
    cl.train()

    # PARAMETERS
    mu = cl.weights
    sigma = 0.5

    # INIT RBF
    rbf = RBF(mu.shape)
    weights = rbf.initWeights()
    print('Weights shape: ', weights.shape)

    # TRAIN RBF
    weights, error_vec, epoch_vec = rbf.train_DELTA(x_train=x_train,
                                                    y_train=y_train,
                                                    weights=weights,
                                                    mu=mu,
                                                    sigma=sigma)
    plt.figure('Training Curve')
    plt.plot(epoch_vec, error_vec)
    plt.title('Training Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Residual Error')

    y_test = rbf.evaluation_DELTA(x_test, weights, mu, sigma)
    print('x: ', y_true[:,0])
    print('y: ', y_true[:,1])
    plt.figure('Delta Rule')
    plt.scatter(y_true[:,0], y_true[:,1], label='True value')
    plt.scatter(y_test[:,0], y_test[:,1], label='Approximation')
    plt.title('Delta Rule')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()



