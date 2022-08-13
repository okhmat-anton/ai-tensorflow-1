import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
import matplotlib.pyplot as plt
from os.path import exists

from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D

print('load data')
(numbers_train, labels_train), (numbers_test, labels_test) = tf.keras.datasets.mnist.load_data()
modelPath = 'sequential-models/sequential-1'


def display_some_examples(examples, labels):
    plt.figure(figsize=(10, 10))
    for i in range(10):
        id = np.random.randint(0, examples.shape[0] - 1)
        img = examples[id]
        label = labels[id]

        plt.subplot(5, 5, i + 1)
        plt.title(str(label))
        plt.tight_layout()
        plt.imshow(img, cmap="gray")

    plt.show()


# display_some_examples(x_train, y_train)

numbers_train = numbers_train.astype('float32') / 255
numbers_test = numbers_test.astype('float32') / 255

numbers_train = np.expand_dims(numbers_train, axis=-1)
numbers_test = np.expand_dims(numbers_test, axis=-1)


if exists(modelPath):
    print('loaded model from '+modelPath+'/model.json')
    model = tfjs.converters.load_keras_model(modelPath+'/model.json')
else:
    # https://medium.com/@samdrinkswater/sequential-model-for-mnist-dataset-using-tensorflow-25e1fab87b48
    model = tf.keras.Sequential(
        [
            Input(shape=(28, 28, 1)),
            Conv2D(32, (3, 3), activation='relu'),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPool2D(),
            BatchNormalization(),

            Conv2D(128, (3, 3), activation='relu'),
            MaxPool2D(),
            BatchNormalization(),

            GlobalAvgPool2D(),
            Dense(64, activation='relu'),
            Dense(10, activation='softmax'),
        ]
    )


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
# model.fit(numbers_train, labels_train, batch_size=64, epochs=1, validation_split=0.2)
model.evaluate(numbers_test, labels_test, batch_size=64)

#tfjs.converters.save_keras_model(model, modelPath)