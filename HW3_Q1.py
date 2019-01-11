# https://www.researchgate.net/publication/306885833_Densely_Connected_Convolutional_Networks
# https://www.youtube.com/watch?v=-W6y8xnd--U
# https://arxiv.org/pdf/1608.06993.pdf
# https://github.com/liuzhuang13/DenseNet

import keras
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, Activation, AveragePooling2D, GlobalAveragePooling2D, concatenate
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import optimizers, regularizers
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf

# Parameters define
growth_rate = 12
depth = 20
compression = 0.5

img_rows, img_cols = 32, 32
img_channels = 3
num_classes = 10
batch_size = 64
epochs = 250
iterations = 782
weight_decay = 1e-4
learning_rate = 0.1
lr_decay = 1e-6
lr_drop = 20

# tensorflow backend
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def lr_scheduler(epoch):
    return learning_rate * (0.5 ** (epoch // lr_drop))


def plot_history(*histories):
    plt.figure(figsize=(20, 10))
    for history, name in histories:
        his = history.history
        val_acc = his['val_acc']
        train_acc = his['acc']
        plt.plot(np.arange(len(val_acc)), val_acc, label=f'{name} val_acc')
        plt.plot(np.arange(len(train_acc)), train_acc, label=f'{name} acc')
        plt.legend()


def normalize(X_train, X_test):
    """
    This function normalize inputs for zero mean and unit variance
    it is used when training a model.
    Input: training set and test set
    Output: normalized training set and test set according to the
    training set statistics.
    """
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train-mean)/(std+1e-7)
    X_test = (X_test-mean)/(std+1e-7)
    return X_train, X_test


def ido_matan_net(x_input, classes_num):
    def conv(x, out_filters, k_size):
        return Conv2D(filters=out_filters,
                      kernel_size=k_size,
                      strides=(1, 1),
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(weight_decay),
                      use_bias=False)(x)

    def dense_layer(x):
        return Dense(units=classes_num,
                     activation='softmax',
                     kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(weight_decay))(x)

    def bn_relu(x):
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        return x

    def bottleneck(x):
        channels = growth_rate * 4
        x = bn_relu(x)
        x = conv(x, channels, (1, 1))
        x = bn_relu(x)
        x = conv(x, growth_rate, (3, 3))
        return x

    def transition(x, inchannels):
        outchannels = int(inchannels * compression)
        x = bn_relu(x)
        x = conv(x, outchannels, (1, 1))
        x = AveragePooling2D((2, 2), strides=(2, 2))(x)
        return x, outchannels

    def dense_block(x, blocks, nchannels):
        concat = x
        for i in range(blocks):
            x = bottleneck(concat)
            concat = concatenate([x, concat], axis=-1)
            nchannels += growth_rate
        return concat, nchannels

    nblocks = (depth - 4) // 6
    nchannels = growth_rate * 2

    x = conv(x_input, nchannels, (3, 3))
    x, nchannels = dense_block(x, nblocks, nchannels)
    x, nchannels = transition(x, nchannels)
    x, nchannels = dense_block(x, nblocks, nchannels)
    x, nchannels = transition(x, nchannels)
    x, nchannels = dense_block(x, nblocks, nchannels)
    x = bn_relu(x)
    x = GlobalAveragePooling2D()(x)
    x = dense_layer(x)
    return x


def main():
    # load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train = K.cast_to_floatx(x_train) / 255
    x_test = K.cast_to_floatx(x_test) / 255

    # Normalize data
    x_train, x_test = normalize(x_train, x_test)

    # build net
    img_input = Input(shape=(img_rows, img_cols, img_channels))
    output = ido_matan_net(img_input, num_classes)
    model = Model(img_input, output)

    print(model.summary())

    # set optimizer
    sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # callback
    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(x_train)

    # Training
    ido_matan_net_history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=iterations,
                        epochs=epochs, callbacks=[reduce_lr], validation_data=(x_test, y_test))

    model.save('ido_matan_net.h5')

    plot_history((ido_matan_net_history, 'ido_matan_net'))


if __name__ == '__main__':
    main()
