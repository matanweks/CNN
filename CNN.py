# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
# import keras
#
# model = Sequential()

#
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
#
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

# matplotlib inline
#
cat_y = np.argmax(y_train, 1)
f, ax = plt.subplots()
n, bins, patches = ax.hist(cat_y, bins=range(11), align='left', rwidth=0.5)
ax.set_xticks(bins[:-1])
ax.set_xlabel('Class Id')
ax.set_ylabel('# Samples')
ax.set_title('CIFAR-10 Class Distribution')

fig = plt.figure(figsize=(10., 10.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(10, 10),  # creates 10x10 grid of axes
                 axes_pad=(0.03, 0.1),  # pad between axes in inch
                 )

for i in range(10):
    cls_id = i
    cat_im = X_train[cat_y == cls_id]

    for j in range(10):
        im = cat_im[j]
        im = im.squeeze()
        ax = grid[10 * i + j]
        ax.imshow(im, cmap='gray')
        ax.axis('off')
        ax.grid(False)
        print(j)
plt.show()
# Prepare datasets
# This step contains normalization and reshaping of input.
# For labels, it is important to change number to one-hot vector.
X_train = K.cast_to_floatx(X_train) / 255
X_test = K.cast_to_floatx(X_test) / 255

# data normalization
#normalize - samplewise for CNN and featurewise for fullyconnected
#
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


X_train, X_test = normalize(X_train, X_test)


# def add_vgg_block(model, num_layers, num_filters):
#     for _ in range(num_layers):
#         model.add(Conv2D(num_filters, (3, 3), padding='same', activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#
#
# def build_vgg16_model():
#     model = Sequential()
#     model.add(Conv2D(64, (3, 3), padding='same', input_shape=(224, 224, 3), activation='relu'))
#     model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#
#     add_vgg_block(model, 2, 128)
#     add_vgg_block(model, 3, 256)
#     add_vgg_block(model, 3, 512)
#     add_vgg_block(model, 3, 512)
#
#     model.add(Flatten())
#     model.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(5e-4)))
#     model.add(Dropout(0.5))
#     model.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(5e-4)))
#     model.add(Dropout(0.5))
#     model.add(Dense(1000, activation='softmax'))
#
#     return model
#
#
# vgg16_model = build_vgg16_model()
# vgg16_model.compile(
#     loss='categorical_crossentropy',
#     optimizer='sgd',
#     metrics=['accuracy']
# )
#
# vgg16_model.summary()