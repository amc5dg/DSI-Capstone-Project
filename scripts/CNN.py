from __future__ import absolute_import
from __future__ import print_function
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, MaxoutDense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from data_cleaning import *

'''
Galaxy Zoo Net Architecture:
input: 45x45x3 RGB arrays scaled to [0,1]
Nesterov momentum mu=0.9
learning rate eta=0.04, decreasing
batch size 16
dropout 0.5 in dense layers
1) convolutional, 32 features, 6x6 filter (also try 8x8), RELu, weights N(0, 0.01)
2) convolutional, 64 features, 5x5 filter (also try 4x4), RELu, weights N(0, 0.01)
3) convolutional, 128 features, 3x3 filter, RELu, weights N(0, 0.01)
4) convolutional, 128 features, 3x3 filter, RELu, weights N(0, 0.1)
5) dense, 2048 features, maxout(2), weights N(0, 0.001)
6) dense, 2048 features, maxout(2), weights N(0, 0.001)
7) dense, 3, softmax????
'''

def scale_features(X):
    '''
    input: X (np array of any dimensions)
    cast as floats for division, scale between 0 and 1
    output: X (np array of same dimensions)
    '''
    X = X.astype("float32")
    X /= 255
    return X


def convert_targets(targets):
    '''
    input: targets (1D np array of strings)
    output: targets dummified category matrix
    note: targets are indexed as ['elliptical', 'merger', 'spiral']
    '''
    return pd.get_dummies(targets).values


def nn_model(X_train, y_train, X_test, y_test, batch_size = 100, nb_classes = 3, nb_epoch = 3):
    # need to fix docs for X_train and X_test as these should be 3D or 4D arrays
    '''
    input: X_train (4D np array), y_train (1D np array), X_test (4D np array), y_test (1D np array)
    optional: batch_size (int), n_classes (int), n_epochs (int)
    output: tpl (test score, test accuracy)
    '''
    # get number of test and train obs
    n_train, n_test = X_train.shape[0], X_test.shape[0]

    # scale images
    X_train, X_test = scale_features(X_train), scale_features(X_test)

    # reshape images because keras is being picky
    X_train = X_train.reshape(n_train, 60, 60, 3)
    X_test = X_test.reshape(n_test, 60, 60, 3)

    # convert class vectors to binary class matrices
    Y_train, Y_test = convert_targets(y_train), convert_targets(y_test)
    # import pdb; pdb.set_trace()

    # initialize sequential model
    model = Sequential()

    # first convolutional layer and subsequent pooling
    # model.add(Convolution2D(32, 1, 1, border_mode='valid', input_shape=(60, 60, 3), activation='relu', dim_ordering='tf', subsample=(1, 1)))
    model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(60, 60, 3), activation='relu', dim_ordering='tf', init='normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # second convolutional layer and subsequent pooling
    model.add(Convolution2D(64, 5, 5, border_mode='valid', activation='relu', init='normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # third convolutional layer
    model.add(Convolution2D(128, 3, 3, border_mode='valid', activation='relu', init='normal'))

    # fourth convolutional layer and subsequent pooling
    model.add(Convolution2D(128, 3, 3, border_mode='valid', activation='relu', init='normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # flattens images to go into dense layers
    model.add(Flatten())

    # first dense layer
    model.add(MaxoutDense(2048))
    #model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # second dense layer
    model.add(MaxoutDense(2048, init='normal'))
    #model.add(Activation('maxout'))
    model.add(Dropout(0.5))

    # test dense layer, remove
    # model.add(Dense(50, init='normal'))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.25))

    # output layer
    model.add(Dense(3, init='normal'))
    model.add(Activation('softmax'))

    # initializes optimizer
    sgd = SGD(lr=0.04, decay=1e-6, momentum=0.9, nesterov=True)

    # compiles and fits model, computes accuracy
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    # print(np.unique(y_train))
    model.fit(X_train, Y_train, show_accuracy=True, verbose=1, batch_size= batch_size, nb_epoch=nb_epoch, validation_data=(X_test, Y_test))
    return model.evaluate(X_test, Y_test, show_accuracy=True, verbose=1)



if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data()
    # np.random.seed(18)  # for reproducibility

    results = nn_model(X_train, y_train, X_test, y_test)
