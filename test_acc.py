import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model

new_model = tf.keras.models.load_model('my_model.h5')

predictions = new_model.predict(x_test)

print(np.argmax(predictions[0]))


plt.imshow(x_test[0],cmap=plt.cm.binary)
plt.show()