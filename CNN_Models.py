import tensorflow as tf
from keras import Model
from keras.layers import Conv2D,BatchNormalization,Activation,MaxPooling2D,Dropout,Flatten,Dense
from keras.models import Sequential

'''
ResNet network model
'''
class ResnetBlock(Model):
    def __init__(self, filters, strides = 1, residual_path = False):
        super(ResnetBlock,self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv2D(filters, 3, strides = strides, padding = 'same', use_bias = False, data_format = 'channels_last')
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D(filters, 3, strides = 1, padding='same', use_bias = False, data_format = 'channels_last')
        self.b2 = BatchNormalization()

        if residual_path:
            self.down_c1 = Conv2D(filters, 1, strides = strides, padding = 'same', use_bias = False, data_format = 'channels_last')
            self.down_b1 = BatchNormalization()

        self.a2 = Activation('relu')

    def call(self, inputs):
        residual = inputs

        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)

        out = self.a2(y + residual)
        return out

class ResNet18(Model):
    def __init__(self, block_list, initial_filters = 64):
        super(ResNet18,self).__init__()
        self.num_blocks = len(block_list)
        self.block_list = block_list
        self.out_filters = initial_filters
        self.c1 = Conv2D(self.out_filters, 3, strides = 1, padding = 'same',use_bias=False,data_format='channels_last')
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.blocks = tf.keras.models.Sequential()

        for block_id in range(len(block_list)):
            for layer_id in range(block_list[block_id]):

                if block_id != 0 and layer_id == 0:
                    block = ResnetBlock(self.out_filters,strides=2,residual_path=True)
                else:
                    block = ResnetBlock(self.out_filters,residual_path=False)
                self.blocks.add(block)
            self.out_filters *= 2
        self.p1 = tf.keras.layers.GlobalAveragePooling2D()
        self.f1 = tf.keras.layers.Dense(7,activation='softmax',kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y

def simple_net():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    return model

