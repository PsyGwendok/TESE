import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import csv
from tqdm import tqdm
import pickle
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from PIL import Image

def imagecrop(file,x1,y1,x2,y2):
    img = cv2.imread(file)
    crop_img = img[y1:y2, x1:x2]
    cv2.imwrite("temp.png",crop_img)

IMG_SIZE = 50

DATADIR = r"C:\Users\Psy\Downloads\Data"

CATEGORIES = ["Panthera pardus", "Loxodonta africana"]

#for category in CATEGORIES:  # do leo and ele
#    input_file = csv.DictReader(open("images.csv"))
#    path = os.path.join(DATADIR,category)  # create path to folder
#    for img in os.listdir(path):  # iterate over each image on each folder
#        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
#        IMG_SIZE =100
#        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#        plt.imshow(new_array, cmap='gray')
        #plt.show()

 #       break  # we just want one for now so break
 #   break  #...and one more!

training_data = []
test_data=[]
 
def create_training_data():  
    count = 0
    #csvfile=open('images.csv','r')
    #csvFileArray = []
	
    #for row in csv.reader(csvfile, delimiter = ','):
        #csvFileArray.append(row)
    for category in CATEGORIES:  # do leo and eles
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=leo 1=ele
  
        for img in tqdm(os.listdir(path)):  # iterate over each image per leos and eles
            print(img)
            try:
                count += 1
                #print(count)
                #print(os.path.join(path,img))
                #print(csvFileArray[count][2])
                #imagecrop(os.path.join(path,img),int(csvFileArray[count][2]),int(csvFileArray[count][3]),int(csvFileArray[count][4]),int(csvFileArray[count][5]))
                #img_array = cv2.imread("temp.png" ,cv2.IMREAD_GRAYSCALE)  # convert to array
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
				
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
                
            except Exception as e:  # in the interest in keeping the output clean...
                raise
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))
    #newpath = r"C:\Users\Psy\Downloads\Data\Imagens\PNB02\Panthera pardus"
    #for img in tqdm(os.listdir(newpath)):  # iterate over each image per leos and eles
    #        try:
                #print(count)
                #print(os.path.join(path,img))
                #print(csvFileArray[count][2])
    #            img_array = cv2.imread(os.path.join(newpath,img) ,cv2.IMREAD_GRAYSCALE)
    #            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
    #            training_data.append([new_array, 1])  # add this to our training_data
                
    #        except Exception as e:  # in the interest in keeping the output clean...
    #				raise
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))
create_training_data()
#training_data.append(test_data)
random.shuffle(training_data)
print(len(training_data))
SAMPLE_SIZE = len(training_data)*0.3
val_data = []
num = 0

while num < SAMPLE_SIZE:
    val_data.append(training_data[num])
    num += 1
	
del training_data[0:int(SAMPLE_SIZE)]


print(len(training_data))

print(len(val_data))
    


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

model.add(Dense(4))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(len(X_val))
print(len(y_val))

model.fit(X, y, batch_size=4, epochs=20, validation_data=(X_val,y_val))