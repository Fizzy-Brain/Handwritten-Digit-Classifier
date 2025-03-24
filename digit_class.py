import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
model = tf.keras.models.load_model('mark1.keras')
i_path = input()
img = tf.keras.utils.load_img(i_path, target_size=(28, 28), color_mode='grayscale')
img = tf.keras.utils.img_to_array(img)
img = img.reshape(1, 28, 28, 1)
img = 1-img.astype('float32')/255.0
prediction = model.predict(img)
print(np.argmax(prediction))