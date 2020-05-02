import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import cv2
import dlab
from skimage import transform
import sys
def load(filename):
    
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (100, 100, 3))
    np_image = np.expand_dims(np_image, axis=0)
    
   
    return np_image
def get_data(path):
        data = load_files(path)
        target_labels = np.array(data['target_names'])
        return target_labels
def main():
    print("loading data (takes about a minute) ...")
    train_data='/Users/stephenowen/Desktop/395/final/fiction/fruits-360/Training'
    test_data='/Users/stephenowen/Desktop/395/final/fiction/fruits-360/Test'
    labels = get_data(train_data)
    
    
    print("MODEL STATS:")
    model = Sequential()

    model.add(Conv2D(filters = 16, kernel_size = 2,input_shape=(100,100,3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters = 32,kernel_size = 2,activation= 'relu',padding='same'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters = 64,kernel_size = 2,activation= 'relu',padding='same'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters = 128,kernel_size = 2,activation= 'relu',padding='same'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters = 256,kernel_size = 2,activation= 'relu',padding='same'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(150))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(120,activation = 'softmax'))
    model.summary()
    #model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.load_weights('cnn_from_scratch_fruits_adam.hdf5')
    print("Model Ready...")
    while(True):
        print("path to image? or type quit to stop")
        file = input()
        if file == "quit":
            sys.exit()
        if file[-3:-1]=="jp" or file[-4:-1]=="jpe":
            image = load(file)
            x=model.predict(image)
            score = np.argmax(x)
            percent = np.max(x)*100
            if percent > 80:
                print("I think this is a "+labels[score])
                print("And I'm "+str(percent)+"% sure of it!")
            else:
                print("I'm not too sure what this is, I think it might be a "+labels[score]+", but I'm not too confident in that.")
        else:
            print("Please input the path to either a JPG or JPEG image file")
if __name__ == "__main__":
    main()