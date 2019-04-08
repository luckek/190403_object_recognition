import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ssig
import scipy.ndimage as snd
import keras.datasets as kd
import keras.models as km
import keras.layers as kl
import keras
import sys


def example_1():
    hedgehog = plt.imread('what_am_i.jpg')
    hedgehog = hedgehog.mean(axis=2)

    sobol_x = np.array([[-1, 0, 1],
                        [-2, 0, 1],
                        [-1, 0, 1]])

    fmap_sobol_x = ssig.convolve2d(hedgehog, sobol_x, mode='valid')
    plt.imshow(fmap_sobol_x)
    plt.show()


def example_2():
    hedgehog = plt.imread('what_am_i.jpg')
    hedgehog = hedgehog.mean(axis=2) / 255.

    sobol_x = 1.0 * np.array([[-1, 0, 1],
                              [-2, 0, 1],
                              [-1, 0, 1]])

    sobol_y = sobol_x.T

    gauss = 1. / 8 * np.array([[0, 1, 0],
                               [1, 2, 1],
                               [0, 1, 0]])

    fmap_sobol_x = ssig.convolve2d(hedgehog, sobol_x, mode='valid')
    fmap_sobol_y = ssig.convolve2d(hedgehog, sobol_y, mode='valid')
    fmap_gauss = ssig.convolve2d(hedgehog, gauss, mode='valid')

    y = relu(np.dstack((fmap_sobol_x, fmap_sobol_y, fmap_gauss)))

    plt.imshow(y)
    plt.show()


def relu(X):
    return np.maximum(X, 0)


def take_subset_based_on_labels(X, y, indices_to_keep):
    boolean_mask = np.logical_or.reduce([y == i for i in indices_to_keep]).squeeze()
    for i, j in enumerate(indices_to_keep):
        y[y == j] = i
    return X[boolean_mask] / 255., y[boolean_mask]


def build_model_orig(input_shape, output_shape):
    model = km.Sequential()

    conv_1 = kl.Conv2D(32, (3, 3), padding='same', input_shape=input_shape, name='conv_1')
    model.add(conv_1)

    act_1 = kl.Activation('relu', name='act_1')
    model.add(act_1)

    bn_1 = kl.BatchNormalization(name='bn_1')
    model.add(bn_1)

    mp_1 = kl.MaxPooling2D(pool_size=(2, 2), name='mp_1')
    model.add(mp_1)

    # Convolution with 64 kernels
    conv_2 = kl.Conv2D(64, (3, 3), padding='same',name='conv_2')
    model.add(conv_2)

    # Activation with ReLU
    act_2 = kl.Activation('relu', name='act_2')
    model.add(act_2)

    # Normalization of output
    bn_2 = kl.BatchNormalization(name='bn_2')
    model.add(bn_2)

    # Downsampling with max pooling
    mp_2 = kl.MaxPooling2D(pool_size=(2, 2), name='mp_2')
    model.add(mp_2)

    # Convolution with 32 kernels
    conv_3 = kl.Conv2D(32, (3, 3), padding='same', name='conv_3')
    model.add(conv_3)

    # Activation with ReLU
    act_3 = kl.Activation('relu', name='act_3')
    model.add(act_3)

    # Normalization of output
    bn_3 = kl.BatchNormalization(name='bn_3')
    model.add(bn_3)

    gap = kl.GlobalAveragePooling2D(name='gap')
    model.add(gap)

    bn_4 = kl.BatchNormalization(name='bn_4')
    model.add(bn_4)

    # first_dense = kl.Dense(output_shape * 2)
    # model.add(first_dense)

    final_dense = kl.Dense(output_shape, name='final_dense')
    model.add(final_dense)

    softmax = kl.Activation('softmax', name='softmax')
    model.add(softmax)

    return model


def copy_model(model):

    bn_3 = model.get_layer(name='bn_3')
    softmax = model.get_layer(name='softmax')

    return km.Model(inputs=model.input, outputs=(bn_3.output, softmax.output))


def main(argv):

    # Set to true to use random subset
    subset_labels = False

    (x_train, y_train), (x_test, y_test) = kd.cifar10.load_data()
    labels = np.array(['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

    new_labels = labels

    # Can take random subset of labels to speed up training /
    # at the cost of reducing generalization accuracy
    if subset_labels:

        indices_to_keep = np.random.choice(len(labels), 3, replace=False)

        print("Keeping indices", indices_to_keep)

        new_labels = [labels[i] for i in indices_to_keep]
        x_train, y_train = take_subset_based_on_labels(x_train, y_train, indices_to_keep)
        x_test, y_test = take_subset_based_on_labels(x_test, y_test, indices_to_keep)

    # Convert class vectors to binary class matrices.
    N = len(new_labels)

    y_train = keras.utils.to_categorical(y_train, N)
    y_test = keras.utils.to_categorical(y_test, N)

    model = build_model_orig(x_train.shape[1:], N)

    # initiate adam optimizer
    opt = keras.optimizers.adam(lr=0.001)

    # model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # TODO: implement save / load routine for weights

    # model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test), shuffle=True)

    final_dense = model.get_layer(name='final_dense')
    print(final_dense.get_weights()[0].shape)

    new_model = copy_model(model)

    # Note that keras expects a batch for predict: our batch here happens to be of size 1
    idx = np.random.randint(0, len(x_train))
    last_conv, probs = new_model.predict(x_train[idx].reshape((1, 32, 32, 3)))

    # Display all 32 feature maps from the final convolutional layer.
    fig, axs = plt.subplots(nrows=8, ncols=4, figsize=(8, 16))
    for j, r in enumerate(axs):
        for i, ax in enumerate(r):
            ax.imshow(last_conv[0, :, :, 4 * j + i])

    print('Probability of each class: ', probs)

    # Make a prediction by taking the argmax of the probabilities
    pred = np.argmax(probs)
    print('This thing is a: ' + new_labels[pred])
    plt.show()

    fm_0 = last_conv[0, :, :, 0]
    fm_0_upscaled = snd.zoom(fm_0, 4)
    print(fm_0.shape, fm_0_upscaled.shape)

    plt.imshow(x_train[idx])
    plt.imshow(fm_0_upscaled, alpha=0.3, cmap='jet')



if __name__ == '__main__':
    main(sys.argv)
