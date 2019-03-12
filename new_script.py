import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import normpath, basename
import cv2
import random
import csv
from tqdm import tqdm
import pickle
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import utils as np_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from PIL import Image
from keras.models import load_model


def imagecrop(file,x1,y1,x2,y2):
    img = cv2.imread(file)
    crop_img = img[y1:y2, x1:x2]	
    cv2.imwrite("temp.png",crop_img)

IMG_SIZE = 100

NAME = "test"

DATADIR = r"C:\Users\Psy\Downloads\Data\Imagens"

CATEGORIES = ["Panthera pardus", "Loxodonta africana", "Sylvicapra grimmia", "Potamochoerus larvatus", "Francolinus sephaena", "Chlorocebus pygerythrus"]

frozen_categories = frozenset(CATEGORIES)

training_data = []
test_data=[]
 
def create_training_data():  
    count = 0
    csvfile=open('images.csv','r')
    csvFileArray = []
	
    for row in csv.reader(csvfile, delimiter = ','):
        csvFileArray.append(row)
		
        
		
    for subdir, dirs, files in os.walk(DATADIR): #category in CATEGORIES:  # do leo and eles
        path = os.path.join(DATADIR,subdir)	
        if (basename(normpath(subdir))) in frozen_categories:
            for file in files:
                print(normpath(subdir))
                class_num = CATEGORIES.index(basename(normpath(subdir)))  # get the classification  (0 or a 1). 0=leo 1=ele
            
                #for img in tqdm(os.listdir(path)):  # iterate over each image per leos and eles
                try:
                    count += 1
                        
                #print(count)
                #print(os.path.join(path,img))
                #print(csvFileArray[count][2])
                #imagecrop(os.path.join(path,img),int(csvFileArray[count][2]),int(csvFileArray[count][3]),int(csvFileArray[count][4]),int(csvFileArray[count][5]))
                #print(os.path.join(path,img))
                #img_array = cv2.imread("temp.png" ,cv2.IMREAD_GRAYSCALE)  # convert to array
                    img_array = cv2.imread(os.path.join(path,file) ,cv2.IMREAD_GRAYSCALE)
                    height, width = img_array.shape
                #print("height:", height,"width:", width)
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                    training_data.append([new_array, class_num])  # add this to our training_data
                
                except Exception as e:  # in the interest in keeping the output clean...
                    raise
           
create_training_data()
#training_data.append(test_data)
random.shuffle(training_data)
#print(len(training_data))
SAMPLE_SIZE = len(training_data)*0.3
val_data = []
num = 0

while num < SAMPLE_SIZE:
    val_data.append(training_data[num])
    num += 1
	
del training_data[0:int(SAMPLE_SIZE)]


#print(len(training_data))

#print(len(val_data))
    


X = []
y = []

X_val = []
y_val = []



for features,label in training_data:
    X.append(features)
    y.append(label)
	
for features,label in val_data:
    X_val.append(features)
    y_val.append(label)

	

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

X_val = np.array(X_val).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

Y = np_utils.to_categorical(y, len(CATEGORIES))
Y_val = np_utils.to_categorical(y_val, len(CATEGORIES))


#print(y.shape)
#print(y_val.shape)
print(X.shape)
print(X_val.shape)


def createModel():
    model = Sequential()

    model.add(Conv2D(16, (3, 3), input_shape=X_val.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors


    model.add(Dense(6))
    model.add(Activation('softmax'))
    return model

#tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

def train():
    model = createModel()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#print(len(X_val))
#print(len(y_val))

    model.fit(X, Y, batch_size=64, epochs=10, validation_data=(X_val,Y_val))

    model.save('my_model.h5')

train()

new_model = tf.keras.models.load_model('my_model.h5')

predictions = new_model.predict(x_test)

print(np.argmax(predictions[0]))


plt.imshow(x_test[0],cmap=plt.cm.binary)
plt.show()

