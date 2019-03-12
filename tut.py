import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np



print(tf.__version__)
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

plt.imshow(x_train[0],cmap=plt.cm.binary)

def create_model():
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
	model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
	model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
	model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
	return model

	
model = create_model()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

			  

model.fit(x_train, y_train, epochs=3)



val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)


model.save_weights("C:/Users/Psy/Data/model.h5")
new_model = create_model()
new_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
new_model.load_weights("C:/Users/Psy/Data/model.h5")

new_model.fit(x_train, y_train, epochs=3)

predictions = new_model.predict(x_test)
print(predictions)		
for x in range(0,10):	  
	print(np.argmax(predictions[x]))
	
for x in range(0,10):
	plt.imshow(x_test[x])
	plt.show()