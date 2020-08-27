import tensorflow as tf
import IPython.display as display
import matplotlib.pyplot as plt


record_file = '../data/dc.record'
raw_image_dataset = tf.data.TFRecordDataset(record_file)
# Create a dictionary describing the features.
image_feature_description = {
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
    return tf.io.parse_single_example(example_proto,image_feature_description)

parsed_image_dataset  = raw_image_dataset.map(_parse_image_function)
print(parsed_image_dataset[0]['image_raw'])




