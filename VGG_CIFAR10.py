import numpy as np
from keras.layers import Dense, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Conv2D, Dropout, Concatenate, Input
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


def build_cifar_10_vgg_model():
    model = Sequential()
    weight_decay = 5e-4

    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=(32, 32, 3), kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.4))
    #
    # model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.4))
    #
    # model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization())
    #
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.4))
    #
    # model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.4))
    #
    # model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization())
    #
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.4))
    #
    # model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.4))
    #
    # model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization())
    #
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model


cifar_10_vgg_model = build_cifar_10_vgg_model()
batch_size = 128
maxepoches = 10
learning_rate = 0.1
lr_decay = 1e-6
lr_drop = 20


def lr_scheduler(epoch):
    return learning_rate * (0.5 ** (epoch // lr_drop))


reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
cifar_10_vgg_model.compile(
    loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=['accuracy']
)
cifar_10_vgg_model.summary()

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

vgg_history = cifar_10_vgg_model.fit_generator(datagen.flow(X_train, y_train,
                                                            batch_size=batch_size),
                                               steps_per_epoch=X_train.shape[0] // batch_size,
                                               epochs=maxepoches,
                                               validation_data=(X_test, y_test),
                                               callbacks=[reduce_lr])
