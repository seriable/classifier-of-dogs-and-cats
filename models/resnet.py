from __future__ import absolute_import,division,print_function
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,Dropout,MaxPooling2D,AvgPool2D,BatchNormalization,Dense,Layer,Flatten

__all__ = ['ResNet18','ResNet34','ResNet50','ResNet101','ResNet152']

class BasicBlock(tf.keras.Model):
    expansion = 1
    def __init__(self,in_planes,planes,stride):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2D(planes, kernel_size=3, strides=(stride, stride), padding='same', use_bias=False)
        self.bn = BatchNormalization()
        self.conv2 = Conv2D(planes, kernel_size=3, strides=(1, 1), padding='same', use_bias=False)
        self.bn2 = BatchNormalization()
        self.shortcut = Sequential()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut.add(Conv2D(self.expansion * planes, kernel_size=1, strides=(stride, stride), use_bias=False))
            self.shortcut.add(BatchNormalization())
        self.relu = tf.keras.activations.relu

    def call(self,inputs):
        out = self.relu(self.bn(self.conv1(inputs)))

        out = self.bn2(self.conv2(out))

        out += self.shortcut(inputs)
        out = self.relu(out)

        return out


class Bottleneck(Layer):
    expansion = 4
    def __init__(self,in_planes,planes,stride=1,):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2D(planes,kernel_size=3,strides=(stride,stride),padding='same',use_bias=False)
        self.bn = BatchNormalization()
        self.conv2 = Conv2D(planes,kernel_size=3,strides=(1,1),padding='same',use_bias=False)

        
        self.conv3 = Conv2D(self.expansion * planes,kernel_size=1,use_bias=False)

        self.shortcut = Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut.add(Conv2D(self.expansion*planes,kernel_size=1,strides=(stride,stride),use_bias=False))
            self.shortcut.add(BatchNormalization())
        self.relu = tf.keras.activations.relu
    
    def call(self,x):
        out = self.relu(self.bn(self.conv1(x)))
        out = self.relu(self.bn(self.conv2(out)))
        out = self.bn(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(tf.keras.Model):
    def __init__(self,block,num_blocks,num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = Conv2D(self.in_planes,kernel_size=3,strides=(1,1),padding='same',use_bias=False)
        self.bn1 = BatchNormalization()
        self.layer1 = self._make_layer(block,64,num_blocks[0],stride=1)
        self.layer2 = self._make_layer(block,128,num_blocks[1],stride=2)
        self.layer3 = self._make_layer(block,256,num_blocks[3],stride=2)
        self.layer4 = self._make_layer(block,512,num_blocks[3],stride=2)
        self.pool = AvgPool2D()
        self.linear = Dense(num_classes)
        self.flatten = Flatten()

        self.relu = tf.keras.activations.relu


    def _make_layer(self,block,planes,num_blocks,stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = Sequential()

        for stride in strides:
            layers.add(block(self.in_planes,planes,stride))
            self.in_planes = planes * block.expansion
        return layers

    def call(self,x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])



if __name__ == '__main__':
    #model = BasicBlock(64,64,2)
    #model = BasicBlock()
    model = ResNet152()
    x = tf.random.normal([2,224,224,3])
    y = model(x)
    print(model.summary())

    print(y)





        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
       