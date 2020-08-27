import  tensorflow as tf
import numpy as np
import cv2
import os


id2label = {0:'cat',1:'dog'}
label2id = {'cat':0,'dog':1}

data_dir = '../data/PetImages/'


# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_label(value):
    if value == 0:
        return tf.train.Feature(FloatList=tf.train.FloatList(value=[1,0]))
    else:
        return tf.train.Feature(FloatList=tf.train.FloatList(value=[0, 1]))


def image_example(image_string,label):
    try:
        image_shape = tf.image.decode_image(image_string).shape
        if image_shape[-1] != 3:
            return 0
    except (tf.python.framework.errors_impl.InvalidArgumentError):
        return 0

    feature = {
      'label': _int64_feature(label),
      'image_raw': _bytes_feature(image_string),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))



record_file = '../data/train.record'

with tf.io.TFRecordWriter(record_file) as writer:
    for pet in os.listdir(data_dir):
        pet_path = os.path.join(data_dir,pet)
        for filename in os.listdir(pet_path):
            if filename.endswith('.jpg') == False:
                continue
            image_path = os.path.join(pet_path,filename)
            #image_rw = tf.io.read_file(image_path)
            try:
                image_rw = open(image_path,'rb').read()
            except IOError:
                print(image_path)
            if image_rw.startswith(b'BM'):
                continue

            try:
                img = np.asarray(image_path)
            except:
                print('corrupt img', image_path)
                continue
            label = label2id.get(pet)
            #print(image_rw)
            print(image_path)
            tf_example = image_example(image_rw,label)
            if tf_example == 0:
                continue
            writer.write(tf_example.SerializeToString())

