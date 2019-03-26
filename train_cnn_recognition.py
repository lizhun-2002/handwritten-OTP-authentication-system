# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 21:28:41 2017

@author: LZ

Train on images split into directories. 

Based on:
https://keras.io/preprocessing/image/
and
https://keras.io/applications/
"""
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

datapath='./data'
#datapath='E:/NIST Special Database 19'
class_num=10#500
img_rows, img_cols = 128, 128
input_shape = (img_rows, img_cols, 3)

# Helper: Save the model.
checkpointer = ModelCheckpoint(
#    filepath='./data/checkpoints/inception.{epoch:03d}-{val_loss:.2f}.hdf5',
    filepath=datapath+'/checkpoints/' + 'cnn' + '-' + 'SDB_19' + '_100epoch.hdf5',
    verbose=1,
    save_best_only=True)

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=3)

# Helper: TensorBoard
tensorboard = TensorBoard(log_dir=datapath+'/logs/')

def get_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        horizontal_flip=True,
        rotation_range=10.,
        width_shift_range=0.2,
        height_shift_range=0.2)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        datapath+'/train',
        target_size=(128, 128),
#        color_mode='grayscale',
        batch_size=256,
#        classes=data.classes,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        datapath+'/test',
        target_size=(128, 128),
#        color_mode='grayscale',
        batch_size=256,
#        classes=data.classes,
        class_mode='categorical')

    return train_generator, validation_generator

def get_model(class_num, input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
#    model = Sequential()
#    model.add(Conv2D(32, kernel_size=(3, 3),
#                     activation='relu',
#                     input_shape=input_shape))
#    model.add(Conv2D(64, (3, 3), activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.25))
#    model.add(Flatten())
#    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(class_num, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model

def train_model(model, nb_epoch, generators, callbacks=[]):
    train_generator, validation_generator = generators
    model.fit_generator(
        train_generator,
        steps_per_epoch=1000,
        validation_data=validation_generator,
        validation_steps=100,
        epochs=nb_epoch,
        callbacks=callbacks)
    return model

def main():
    model = get_model(class_num, input_shape)
    generators = get_generators()

    model = train_model(model, 1000, generators,
#                                callbacks=[early_stopper])
                        [checkpointer, early_stopper, tensorboard])

if __name__ == '__main__':
    main()
