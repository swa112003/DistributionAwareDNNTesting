'''
#Train a deep CNN on the MNIST images dataset.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dropout, Activation, Input, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.initializers import glorot_normal, RandomNormal, Zeros
import os
import time
import numpy as np


def two_conv_pool(x, F1, F2, name):
        x = Conv2D(F1, (3, 3), activation=None, padding='valid', name='{}_conv1'.format(name))(x)
        x = Activation('relu')(x)
        x = Conv2D(F2, (3, 3), activation=None, padding='valid', name='{}_conv2'.format(name))(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='{}_pool'.format(name))(x)

        return x
    
        
def Model4(input_tensor=None, train=False):
    img_rows, img_cols, img_chn = 28, 28, 1
    input_shape = (img_rows, img_cols, img_chn)
    if train:
        # start_time = time.clock()
        batch_size = 128
        num_classes = 10
        epochs = 10       
        
        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        # Convert class vectors to binary class matrices.
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)   
        # input image dimensions
        input_tensor = Input(shape=input_shape)

    # Model definition   
    net = two_conv_pool(input_tensor, 32, 32, "block1")
    net = two_conv_pool(net, 64, 64, "block2")

    net = Flatten()(net)
    net = Dense(200, name='fc-1')(net)
    net = Activation('relu')(net)
    net = Dense(200, name='fc-2')(net)
    net = Activation('relu')(net)
    net = Dense(10, name='before_softmax')(net)
    net = Activation('softmax')(net)
    
    model = Model(input_tensor, net) 
    #model.summary()
    if train:
        # compiling
        sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), verbose=2, shuffle=True)
        
        # save model
        model.save_weights('./Model4.h5')
        score = model.evaluate(x_test, y_test, verbose=0)
        print('\n')
        print('Overall Test score:', score[0])
        print('Overall Test accuracy:', score[1])
        # Printing out training execution time
        # print("--- %s seconds ---" % (time.clock() - start_time))
    else:
        model.load_weights('./Model4.h5')
        
        print('MNIST Model4 loaded')
    return model





if __name__ == '__main__':
    Model4(train=True)
