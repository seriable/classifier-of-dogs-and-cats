import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dropout,MaxPooling2D,AvgPool2D,BatchNormalization,Dense,Layer,Flatten



__All__ = ['alexnet']

class AlexNet(tf.keras.Model):
    def __init__(self,num_classes=2):
        super(AlexNet, self).__init__()

        self.conv1 = Conv2D(64,11,strides=(5,5),padding='same',activation='relu',input_shape=(100,100,3))
        self.pool = MaxPooling2D(pool_size=(2,2))
        self.conv2 = Conv2D(192,5,padding='same',strides=(1,1),activation='relu')
        self.conv3 = Conv2D(384,3,padding='same',strides=(1,1),activation='relu')
        self.conv4 = Conv2D(256,3,padding='same',activation='relu')
        self.conv5 = Conv2D(256,3,padding='same',activation='relu')
        self.linear = Dense(num_classes)
        self.flatten = Flatten()
    def call(self,x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(self.conv5(x))
        x = self.flatten(x)
        y = self.linear(x)

        return y
'''

if __name__ == '__main__':
    model = AlexNet()

    x = tf.random.normal(shape=(2,100,100,3))
    y = model(x)


    print(model.summary())
    print(y)

'''

def alexnet(num_classes=2):
    return AlexNet(num_classes=num_classes)
