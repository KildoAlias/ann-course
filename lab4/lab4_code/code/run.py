from util import *
from rbm import RestrictedBoltzmannMachine
from dbn import DeepBeliefNet
import matplotlib.pyplot as plt

if __name__ == "__main__":

    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(
        dim=image_size, n_train=60000, n_test=10000)

    ''' restricted boltzmann machine '''

    print ("\nStarting a Restricted Boltzmann Machine..")

    hidden_vec = [200, 250, 300, 350, 400, 450, 500]
    loss_matrix = []
    for ndim_hidden in hidden_vec:
        rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                         ndim_hidden=ndim_hidden,
                                         is_bottom=True,
                                         image_size=image_size,
                                         is_top=False,
                                         n_labels=10,
                                         batch_size=20
                                         )

        # print ("shape of training data: ",train_imgs.shape)
        loss_vec, it_vec = rbm.cd1(
            visible_trainset=train_imgs, n_iterations=10)
        loss_matrix.append(loss_vec)
        print('ndim hidden Units:{} \t loss value: {}\n'.format(
            ndim_hidden, loss_vec[-1]))

    plt.figure('Learning Curves')
    i = 0
    for loss_vec in loss_matrix:
        plt.plot(it_vec, loss_vec,
                 label='Hidden Units: {}'.format(hidden_vec[i]))
        i += 1
    plt.title('Learning Curve RBM')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    quit()
    ''' deep- belief net '''
    print ("\nStarting a Deep Belief Net..")

    dbn = DeepBeliefNet(sizes={"vis": image_size[0]*image_size[1], "hid": 500, "pen": 500, "top": 2000, "lbl": 10},
                        image_size=image_size,
                        n_labels=10,
                        batch_size=10
                        )

    ''' greedy layer-wise training '''

    dbn.train_greedylayerwise(vis_trainset=train_imgs,
                              lbl_trainset=train_lbls, n_iterations=2000)

    dbn.recognize(train_imgs, train_lbls)

    dbn.recognize(test_imgs, test_lbls)

    for digit in range(10):
        digit_1hot = np.zeros(shape=(1, 10))
        digit_1hot[0, digit] = 1
        dbn.generate(digit_1hot, name="rbms")

    ''' fine-tune wake-sleep training '''

    dbn.train_wakesleep_finetune(
        vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=2000)

    dbn.recognize(train_imgs, train_lbls)

    dbn.recognize(test_imgs, test_lbls)

    for digit in range(10):
        digit_1hot = np.zeros(shape=(1, 10))
        digit_1hot[0, digit] = 1
        dbn.generate(digit_1hot, name="dbn")
