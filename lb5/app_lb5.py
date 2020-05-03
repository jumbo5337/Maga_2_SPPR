import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

lr = 0.01
epochs = 5
batch_size = 64
conv_size = 5
pooling='avg'
activation = 'tanh'
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

test_labels = to_categorical(test_labels)
train_labels = to_categorical(train_labels)



model = models.Sequential()
if conv_size == 5 :
    model.add(layers.Conv2D(32, (5, 5), activation=activation,
                        input_shape=(28, 28, 1)))
elif conv_size == 3 :
    model.add(layers.Conv2D(32, (3, 3), activation=activation,
                            input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(32, (3, 3), activation=activation,
                            input_shape=(28, 28, 1)))
else:
    raise NotImplementedError
model.add(layers.BatchNormalization(axis=1))

if pooling=='avg' :
    model.add(layers.AveragePooling2D((2, 2)))
else :
    model.add(layers.AveragePooling2D((2, 2)))

if conv_size == 5:
    model.add(layers.Conv2D(64, (5, 5), activation=activation))
elif conv_size == 3:
    model.add(layers.Conv2D(64, (3, 3), activation=activation))
    model.add(layers.Conv2D(64, (3, 3), activation=activation))
else:
    raise NotImplementedError
model.add(layers.BatchNormalization(axis=1))
model.add(layers.MaxPooling2D((2, 2)))
if conv_size == 5:
    model.add(layers.Conv2D(128, (5, 5), activation=activation))
elif conv_size == 3:
    model.add(layers.Conv2D(64, (3, 3), activation=activation))
    model.add(layers.Conv2D(64, (3, 3), activation=activation))
else:
    raise NotImplementedError
model.add(layers.BatchNormalization(axis=1))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation=activation))
model.add(layers.Dense(10, activation='softmax'))
ADAM = optimizers.Adam(lr=lr)

model.compile(optimizer=ADAM,
                loss = 'categorical_crossentropy',
                metrics=['accuracy'])


start_time = datetime.now()

history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size)
test_loss, test_acc = model.evaluate(test_images, test_labels)



print(datetime.now()-start_time)

plt.plot(history.history['accuracy'], color ='r', label ='train-accuracy')
plt.hlines(test_acc,0,int(epochs-1),colors = 'b', ls = '--', label='test-accuracy')
plt.legend(loc = 4)
plt.title("Accuracy")
plt.savefig("accuracy.png")
plt.show()
plt.close()

plt.plot(history.history['loss'], color ='r', label ='train-loss')
plt.hlines(test_loss,0,int(epochs-1),colors = 'b', ls = '--', label='test-loss')
plt.legend(loc = 3)
plt.title("Losses")
plt.savefig("loss.png")
plt.show()
plt.close()
print(test_loss)
print(test_acc)



