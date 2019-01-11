import numpy as np
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Conv2D, Concatenate, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers import Flatten, InputLayer, BatchNormalization, Activation
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.initializers import Constant
from keras.datasets import cifar10
import keras.backend as K
from keras.layers.core import Lambda
from keras import regularizers
from keras import optimizers
import keras

from sklearn.model_selection import train_test_split


num_classes = 10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)


# Prepare datasets
# This step contains normalization and reshaping of input.
# For labels, it is important to change number to one-hot vector.
X_train = K.cast_to_floatx(X_train) / 255
X_test = K.cast_to_floatx(X_test) / 255

#data normalization
#normalize - samplewise for CNN and featurewise for fullyconnected

def normalize(X_train,X_test):
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


X_train,X_test = normalize(X_train,X_test)


def build_cifar_10_googlenet(l2=5e-4):
    x_in = Input(shape=(32, 32, 3))

    # x = Conv2D(64, kernel_size=(7,7), strides=(2,2), activation='relu', padding='same')(x_in)
    # x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
    # skipping LRM here for simplicity
    x = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2))(x_in)
    x = BatchNormalization()(x)
    x = Conv2D(192, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2))(x)
    x = BatchNormalization()(x)
    # skipping LRM here for simplicity
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = inception_module(x, 64, 96, 128, 16, 32, 32, l2=l2)
    x = BatchNormalization()(x)
    x = inception_module(x, 128, 128, 192, 32, 96, 64, l2=l2)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = inception_module(x, 192, 96, 208, 16, 48, 64, l2=l2)
    x = BatchNormalization()(x)
    x = inception_module(x, 160, 112, 224, 24, 64, 64, l2=l2)
    x = BatchNormalization()(x)
    x = inception_module(x, 128, 128, 256, 24, 64, 64, l2=l2)
    x = BatchNormalization()(x)
    x = inception_module(x, 112, 144, 288, 32, 64, 64, l2=l2)
    x = BatchNormalization()(x)
    x = inception_module(x, 256, 160, 320, 32, 128, 128, l2=l2)
    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = inception_module(x, 256, 160, 320, 32, 128, 128, l2=l2)
    x = BatchNormalization()(x)
    x = inception_module(x, 384, 192, 384, 48, 128, 128, l2=l2)
    x = BatchNormalization()(x)

    x = AveragePooling2D(pool_size=(4, 4), strides=(2, 2), padding='valid')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.0002))(x)

    return Model(inputs=x_in, outputs=x)


def inception_module(x_in, num_1_filters, num_3_reduce, num_3_filters, num_5_reduce, num_5_filters, num_pool_proj,
                     l2=0.):
    conv1x1 = Conv2D(num_1_filters, kernel_size=(1, 1), padding='same', activation='relu', strides=(1, 1),
                     kernel_regularizer=regularizers.l2(l2))(x_in)

    conv3x3_reduce = Conv2D(num_3_reduce, kernel_size=(1, 1), padding='same', activation='relu', strides=(1, 1),
                            kernel_regularizer=regularizers.l2(l2))(x_in)
    conv3x3 = Conv2D(num_3_filters, kernel_size=(3, 3), padding='same', activation='relu', strides=(1, 1),
                     kernel_regularizer=regularizers.l2(l2))(conv3x3_reduce)

    conv5x5_reduce = Conv2D(num_5_reduce, kernel_size=(1, 1), padding='same', activation='relu', strides=(1, 1),
                            kernel_regularizer=regularizers.l2(l2))(x_in)
    conv5x5 = Conv2D(num_5_filters, kernel_size=(5, 5), padding='same', activation='relu', strides=(1, 1),
                     kernel_regularizer=regularizers.l2(l2))(conv5x5_reduce)

    maxpool3x3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x_in)
    pool_proj = Conv2D(num_pool_proj, kernel_size=(1, 1), padding='same', activation='relu', strides=(1, 1),
                       kernel_regularizer=regularizers.l2(l2))(maxpool3x3)

    return Concatenate(axis=3)([conv1x1, conv3x3, conv5x5, pool_proj])


def aux_classifier(x_in):
    aux = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x_in)
    aux = Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(aux)
    aux = Flatten()(aux)
    aux = Dense(1024, activation='relu')(aux)
    aux = Dropout(0.7)(aux)
    return Dense(1000, activation='softmax')(aux)


def build_googlenet(aux_loss=False):
    x_in = Input(shape=(224, 224, 3))

    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), activation='relu', padding='same')(x_in)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    # skipping LRM here for simplicity
    x = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(192, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    # skipping LRM here for simplicity
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = inception_module(x, 64, 96, 128, 16, 32, 32)
    x = inception_module(x, 128, 128, 192, 32, 96, 64)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = inception_module(x, 192, 96, 208, 16, 48, 64)
    if aux_loss:
        aux1 = aux_classifier(x)

    x = inception_module(x, 160, 112, 224, 24, 64, 64)
    x = inception_module(x, 128, 128, 256, 24, 64, 64)
    x = inception_module(x, 112, 144, 288, 32, 64, 64)

    x = inception_module(x, 256, 160, 320, 32, 128, 128)
    if aux_loss:
        aux2 = aux_classifier(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = inception_module(x, 256, 160, 320, 32, 128, 128)
    x = inception_module(x, 384, 192, 384, 48, 128, 128)

    x = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid')(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(1000, activation='softmax')(x)

    if aux_loss:
        return Model(inputs=x_in, outputs=[x, aux1, aux2])
    return Model(inputs=x_in, outputs=x)


# use_aux = False
# googlenet_model = build_googlenet(use_aux)
#
# googlenet_model.compile(
#     loss='categorical_crossentropy',
#     optimizer=sgd,
#     metrics=['accuracy'],
#     loss_weights=[1., 0.3, 0.3] if use_aux else None
# )
# googlenet_model.summary()


cifar_10_googlenet = build_cifar_10_googlenet(l2=0.0002)
batch_size = 128
maxepoches = 250
learning_rate = 0.01
lr_decay = 1e-6
lr_drop = 20


def lr_scheduler(epoch):
    return learning_rate * (0.5 ** (epoch // lr_drop))


reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
cifar_10_googlenet.compile(
    loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=['accuracy']
)
cifar_10_googlenet.summary()

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
datagen.fit(X_train)

googlenet_history = cifar_10_googlenet.fit_generator(datagen.flow(X_train, y_train,
                                                                  batch_size=batch_size),
                                                     steps_per_epoch=X_train.shape[0] // batch_size,
                                                     epochs=maxepoches,
                                                     validation_data=(X_test, y_test),
                                                     callbacks=[reduce_lr])
plot_history((googlenet_history, 'googlenet'))