import tensorflow as tf
import models

def preprocess_image(image):
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image,[224,224])
    image /= 255.0  # normalize to [0,1] range
    return image


record_file = './data/train.record'

raw_image_dataset = tf.data.TFRecordDataset(record_file)
# Create a dictionary describing the features.
image_feature_description = {
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
    feature =  tf.io.parse_single_example(example_proto,image_feature_description)
    image = tf.reshape(tf.image.resize(tf.image.decode_jpeg(feature['image_raw']),[100,100]),[100,100,3])/255.0
    label = feature['label']
    #return image,[1,0] if label==1 else [0,1]
    return image,label

parsed_image_dataset  = raw_image_dataset.map(_parse_image_function)

'''
i = 0
for image_features in parsed_image_dataset:
    label = image_features['label']
    print(label)

train_x = [tf.image.resize(tf.image.decode_jpeg(image_features['image_raw']),[100,100]) for image_features in parsed_image_dataset ]
train_y = [[1,0] if image_features['label']==0 else  [0,1] for image_features in parsed_image_dataset  ]

dataset = parsed_image_dataset.shuffle(buffer_size=1024).batch(32)
'''

#model = models.alexnet()
model = models.ResNet18()
dataset = parsed_image_dataset.shuffle(buffer_size=20000).batch(20).repeat(1)

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])
model.fit(dataset,epochs=1,steps_per_epoch=1200)
model.save('./model/dc')
'''
optimizer = tf.keras.optimizers.Adam(0.01)

def loss_fn(y_true,y_pred):
    loss = tf.math.subtract(y_true,y_pred)
    return loss

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

for image,label in dataset:
    with tf.GradientTape() as tape:
        y_pred = model(image)
        loss_value = loss_object(label,y_pred)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print("-" * 100)
    print(loss_value.numpy())
'''










