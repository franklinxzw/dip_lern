import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Sequential
from tensorflow.keras import Model


# One or two conv layers with same # of filters and then a pooling
class ConvBlock(Model):
	def __init__(self, filters, repeats):
    super(Block, self).__init__()
    self.convs = Sequential([Conv2D(filters, 3, activation='relu') for _ in repeats])
    self.pool = MaxPool2D(2, 2, padding='none')

  def call(self, x):
    return self.pool(self.convs(x))  

class VGG11(Model):
      def __init__(self, n_classes=10):
        super(VGG11, self).__init__()

        self.n_classes = n_classes
        self.conv_blocks = Sequential()
        for n_filters, reps in zip([64, 128, 256, 512], [1, 1, 2, 2]):
            self.conv_blocks.add(Convblock(filters=n_filters, repeats=reps))
            
        self.flatten = Flatten()
        self.fc_512_a = Dense(512, activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
        self.fc_512_b = Dense(512, activation='relu', use_bias=True, kernel_initializer='glorot_uniform')
        self.fc_100_softmax = Dense(n_classes, activation='softmax', use_bias=True, kernel_initializer='glorot_uniform')
		
  def call(self, x):
    x = self.conv_blocks(x)
    x = self.flatten(x)
    x = self.fc_512_a(x)
    x = self.fc_512_b(x)
    return self.fc_100_softmax(x)