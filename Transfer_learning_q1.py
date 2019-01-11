from __future__ import print_function
import keras
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.layers import *
from keras.utils import plot_model
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


class cifar100vgg:
    def __init__(self, train=True):
        self.num_classes = 10
        self.weight_decay = 0.0005
        self.x_shape = [32, 32, 3]

        self.model = self.build_model()
        if train:
            self.model = self.train(self.model)
        else:
            self.model.load_weights('cifar100vgg.h5', by_name=True)

    def build_model(self):
        # Build the network for EKNN (Embedded K Nearest Neighbors)

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))

        return model


    def normalize(self,X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        print(mean)
        print(std)
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test

    def normalize_production(self,x):
        #this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        #these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 121.936
        std = 68.389
        return (x-mean)/(std+1e-7)

    def predict(self, x, normalize=True, batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x, batch_size)

    def train(self, model):

        #training parameters
        batch_size = 128
        maxepoches = 250
        epochs = 40
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 20

        # The data, shuffled and split between train and test sets:
        # (x_train, y_train), (x_test, y_test) = cifar100.load_data()

        (x_train_full, y_train_full), (x_test_full, y_test_full) = cifar10.load_data()

        x_train, _, y_train, _ = train_test_split(x_train_full, y_train_full, train_size=500, random_state=42,
                                                              stratify=y_train_full)

        x_test, _, y_test, _ = train_test_split(x_test_full, y_test_full, train_size=100, random_state=42,
                                                              stratify=y_test_full)

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)


        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)


        #data augmentation
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
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)


        #optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


        # training process in a for loop with learning rate drop every 25 epoches.

        history = model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=maxepoches,
                            validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=2)


        # history = model.fit(
        #     x_train, y_train,
        #     batch_size=batch_size,
        #     epochs=epochs,
        #     verbose=1,
        #     validation_data=(x_test, y_test))

        # predicted_x = model.predict(x_test_full)
        # residuals = (np.argmax(predicted_x, 1) != np.argmax(y_test_full, 1))
        # loss = sum(residuals) / len(residuals)
        # print("the validation 0/1 loss is: ", loss)

        his = history.history
        x = list(range(epochs))
        y_1 = his['val_acc']
        y_2 = his['acc']
        plt.plot(x, y_1)
        plt.plot(x, y_2)
        plt.legend(['validation accuracy', 'training_accuracy'])
        plt.show()

        model.save_weights('cifar10vgg_bo100.h5')
        return model


def print_result(y, y_bar):
    x_axis = np.linspace(1, len(y), len(y))
    # ls = plt.figure(1)
    plt.plot(x_axis, y, 'r')
    plt.plot(x_axis, y_bar, 'b')
    plt.legend(["Y", "Y_bar"])
    plt.ylabel("label")
    plt.xlabel("sample")
    plt.title('Label Plot')
    plt.show()


def load_data(train_size, test_size):

    (x_train_full, y_train_full), (x_test_full, y_test_full) = cifar10.load_data()

    x_train, _, y_train, _ = train_test_split(x_train_full, y_train_full, train_size=train_size, random_state=42,
                                                                  stratify=y_train_full)
    x_test, _, y_test, _ = train_test_split(x_test_full, y_test_full, train_size=test_size, random_state=42,
                                                                  stratify=y_test_full)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train, x_test = VGG10_EKNN.normalize(x_train, x_test)

    return x_train, y_train, x_test, y_test


def KNN_loss(y_test, y_bar):

    residuals_acc = (y_bar == y_test.reshape(-1))
    acc = sum(residuals_acc) / len(residuals_acc)

    return acc


def predict_KNN(k, features, y_train, x_test):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(features, np.ravel(y_train))

    test_set = VGG10_EKNN.model.predict(x_test)

    y_bar = neigh.predict(test_set)
    return np.round(y_bar)


VGG10_EKNN = cifar100vgg(train=False)

VGG10_EKNN.model.summary()

[x_train, y_train, x_test, y_test] = load_data(train_size=1000, test_size=1000)

features = VGG10_EKNN.model.predict(x_train)

acc = []

for k in range(1, 40):

    y_bar = predict_KNN(k, features, y_train, x_test)
    accuracy = KNN_loss(y_test, y_bar)
    acc.append(accuracy)
    print('The validation 0/1 accuracy for k= ' + str(k) + ' is: ', str(accuracy))

plt.plot(np.arange(len(acc)), acc)
plt.ylabel("0/1 accuracy")
plt.xlabel("K")
plt.title('Accuracy vs K')
plt.show()






