import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
import cv2


DATADIR = r"C:\Users\Psy\Downloads\Data\Imagens\PNB02\Loxodonta africana\PNB02__2017-09-14__21-48-00(1)__Loxodonta africana.JPG" 


def valid_imshow_data(data):
    data = np.asarray(data)
    if data.ndim == 2:
        return True
    elif data.ndim == 3:
        if 3 <= data.shape[2] <= 4:
            return True
        else:
            print('The "data" has 3 dimensions but the last dimension '
                  'must have a length of 3 (RGB) or 4 (RGBA), not "{}".'
                  ''.format(data.shape[2]))
            return False
    else:
        print('To visualize an image the data must be 2 dimensional or '
              '3 dimensional, not "{}".'
              ''.format(data.ndim))
        return False


#for subdir, dirs, files in os.walk(DATADIR)
 #   path = C:\Users\Psy\Downloads\Data\Imagens\PNB05\Loxodonta africana\PNB05__2018-05-08__01-03-39(24)__Loxodonta africana.PNG"


	
#arranjar o file

new_model = tf.keras.models.load_model('my_model.h5')
X_val = cv2.imread(DATADIR ,cv2.IMREAD_GRAYSCALE)
#height, width = X_val.shape
#print("height:", height,"width:", width)	
new_array=cv2.resize(X_val, (100, 100))
X_val = np.array(new_array).reshape(-1, 100, 100, 1)


predictions = new_model.predict(X_val)

print(np.argmax(predictions[0]))


new_SN_map = np.array(X_val)
valid_imshow_data(new_SN_map)

#X_val = X_val.squeeze()
plt.imshow(X_val[0].squeeze())
plt.show()