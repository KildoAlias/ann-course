import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers


def generateTimeseries():
    x = 1.5
    x_next = []

    for i in range(25):
        x_next.append(x)
        x *= 0.9

    for i in range(1600):
        x_next.append(x_next[-1] + (0.2*x_next[-25]) /
                      (1+x_next[-25]**10) - 0.1 * x_next[-1])

    return x_next


if __name__ == "__main__":

    plot_t = np.linspace(1, 1500, 1500)

    x = generateTimeseries()
    x = np.array(x)

    t_vec = np.linspace(301, 1499, 1199)

    print(t_vec.shape)
    x_in = np.array([[x[300-20]], [x[300-15]],
                     [x[300-10]], [x[300-5]], [x[300]]])

    out = np.array([x[300+5]])

    for t in t_vec:
        t = int(t)
        x_temp = np.array([[x[t-20]], [x[t-15]],
                           [x[t-10]], [x[t-5]], [x[t]]])
        x_in = np.hstack((x_in, x_temp))

        out_temp = np.array(x[t+5])
        out = np.hstack((out, out_temp))

    x_train = np.transpose(x_in[:, :1000])
    y_train = out[:1000]
    x_test = np.transpose(x_in[:, 1000:])
    y_test = out[1000:]

    # Create the NN
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(5, )))
    model.add(tf.keras.layers.Dense(32, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(1))

    optimizer = keras.optimizers.Adam(learning_rate=0.1)
    model.compile(optimizer=optimizer,
                  loss='mse')

    print("Fit model on training data")
    callback = [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=1)
    ]

    history = model.fit(
        x_train,
        y_train,
        shuffle=False,
        validation_split=0.3,
        batch_size=64,
        epochs=2000,
        callbacks=callback)

    plt.figure("Learning Curve")
    plt.plot(history.history['loss'], color='k')
    plt.plot(history.history['val_loss'], color='r')
    plt.title('Learning Curves')
    plt.legend(['loss', 'val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('MSE')

    loss_value = model.evaluate(x_test, y_test)
    prediction = model.predict(x_test)
    plt.figure('Prediction')
    plt.plot(prediction)
    plt.plot(y_test)
    plt.show()
