import tensorflow as tf
import tensorflow.keras as keras
import models

#model = models.ResNet34()
#model.load_weights('./model/dc')
model = keras.models.load_model('./model/dc')

#model = keras.models.load_model('./model/dc')
#print(model.summary())
image = tf.reshape(tf.image.resize(tf.image.decode_jpeg(tf.io.read_file('./101.jpg')),[100,100]),[1,100,100,3])
print(image.shape)
print(model(image))