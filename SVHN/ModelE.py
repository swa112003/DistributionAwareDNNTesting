'''
#Train a deep CNN on the SVHN images dataset.

Model from below paper VGG-19
K. Simonyan and A. Zisserman. Very deep convolutional networks for
large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014
'''

from __future__ import print_function
import keras
from scipy.io import loadmat
from keras.models import Model
from keras.layers import Dropout, Activation, Input, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import os
import time
import numpy as np


lr_drop = 20
learning_rate = 0.1        
        
def ModelE(input_tensor=None, train=False):
    img_rows, img_cols, img_chn = 32, 32, 3
    input_shape = (img_rows, img_cols, img_chn)
    if train:
        # start_time = time.clock()
        batch_size = 128
        num_classes = 10
        epochs = 100        
        
        datasetLoc = './dataset/'
        train_data = loadmat(datasetLoc+'train_32x32.mat')
        x_train = np.array(train_data['X'])
        y_train = train_data['y']
        test_data = loadmat(datasetLoc+'test_32x32.mat')
        x_test = np.array(test_data['X'])
        y_test = test_data['y']

        x_train = np.moveaxis(x_train, -1, 0)
        x_test = np.moveaxis(x_test, -1, 0)
        # Normalize data.
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        
        y_train[y_train == 10] = 0
        y_train = np.array(y_train)
        y_test[y_test == 10] = 0
        y_test = np.array(y_test)

        # Convert class vectors to binary class matrices.
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)   
        # input image dimensions
        input_tensor = Input(shape=input_shape)
        
    # Model definition   
    x = Conv2D(64, (3, 3), input_shape=input_shape, padding='same')(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(4096)(x)
    x = Activation('relu')(x)
    x = Dense(4096)(x)
    x = Activation('relu')(x)
    x = Dense(10, name='before_softmax')(x)
    x = Activation('softmax')(x)
    
    model = Model(input_tensor, x) 
    #model.summary()
    if train:
        # compiling
        filepath="ModelE.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=2, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        model.fit(x_train, y_train, callbacks=callbacks_list, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), verbose=2, shuffle=True)
        
        # save model
        model.save_weights('./ModelE.h5')
        score = model.evaluate(x_test, y_test, verbose=0)
        print('\n')
        print('Overall Test score:', score[0])
        print('Overall Test accuracy:', score[1])
        # Printing out training execution time
        # print("--- %s seconds ---" % (time.clock() - start_time))
    else:
        model.load_weights('./ModelE.h5')
        
        print('SVHN ModelE loaded')
    return model





if __name__ == '__main__':
    ModelE(train=True)
