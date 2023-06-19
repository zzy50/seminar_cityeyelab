import os
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.datasets import cifar10
from keras.callbacks import TensorBoard
from pathlib import Path
from datetime import datetime
 
"""
구현할 목록
1. activation func(relu, softmax)
2. 2D Convolution
3. BatchNormalization
4. 2D MaxPooling
5. Flatten
6. Dense with activation func(Fully Connected Layer)
7. Dropout
8. Cost func (sparse_categorical_crossentropy)
9. Opimizer (SGD)
"""

# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. 
# There are 50000 training images and 10000 test images.
CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


class AlexNet(Sequential):
    def __init__(self):
        super().__init__()
        self.add(layers.Conv2D(filters=96, kernel_size=(11, 11), 
                                strides=(4, 4), activation="relu", 
                                input_shape=(227, 227, 3)))
        self.add(layers.BatchNormalization())
        self.add(layers.MaxPool2D(pool_size=(3, 3), strides= (2, 2)))
        self.add(layers.Conv2D(filters=256, kernel_size=(5, 5), 
                                strides=(1, 1), activation="relu", 
                                padding="same"))
        self.add(layers.BatchNormalization())
        self.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
        self.add(layers.Conv2D(filters=384, kernel_size=(3, 3), 
                                strides=(1, 1), activation="relu", 
                                padding="same"))
        self.add(layers.BatchNormalization())
        self.add(layers.Conv2D(filters=384, kernel_size=(3, 3), 
                                strides=(1, 1), activation="relu", 
                                padding="same"))
        self.add(layers.BatchNormalization())
        self.add(layers.Conv2D(filters=256, kernel_size=(3, 3), 
                                strides=(1, 1), activation="relu", 
                                padding="same"))
        self.add(layers.BatchNormalization())
        self.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
        self.add(layers.Flatten())

        self.add(layers.Dense(4096, activation="relu"))
        self.add(layers.Dropout(0.5))
        self.add(layers.Dense(4096, activation="relu"))
        self.add(layers.Dropout(0.5))
        self.add(layers.Dense(1000, activation="softmax"))
        
        self.compile(loss='sparse_categorical_crossentropy', 
                    optimizer=tf.optimizers.SGD(learning_rate=0.0005), 
                    metrics=['accuracy'])





def dataloader():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    validation_images, validation_labels = train_images[:5000], train_labels[:5000]
    train_images, train_labels = train_images[5000:], train_labels[5000:]

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

    train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
    test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
    validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()
    print("Training data size:", train_ds_size)
    print("Test data size:", test_ds_size)
    print("Validation data size:", validation_ds_size)
    
    train_ds = (train_ds
                    .map(process_images).cache()
                    .shuffle(buffer_size=train_ds_size)
                    .batch(batch_size=32, drop_remainder=True))
    test_ds = (test_ds
                    .map(process_images).cache()
                    .shuffle(buffer_size=train_ds_size)
                    .batch(batch_size=32, drop_remainder=True))
    validation_ds = (validation_ds
                    .map(process_images).cache()
                    .shuffle(buffer_size=train_ds_size)
                    .batch(batch_size=32, drop_remainder=True))

    return train_ds, test_ds, validation_ds


def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227,227))
    return image, label


def get_tensorboard_logger():
    now = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    logdir_root = Path("logs") / now
    logdir_root.mkdir(exist_ok=True, parents=True)
    tensorboard_cb = TensorBoard(logdir_root)
    return tensorboard_cb


def train(model: Sequential, train_ds, validation_ds) -> None:
    print("\n========== train & validation ==========")
    tensorboard_logger = get_tensorboard_logger()
    model.fit(train_ds,
            epochs=50,
            validation_data=validation_ds,
            validation_freq=1,
            callbacks=[tensorboard_logger])

def test(model: Sequential, test_ds) -> None:
    print("\n========== test ==========")
    model.evaluate(test_ds)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    with tf.device("/gpu:0"):
        train_ds, test_ds, validation_ds = dataloader()
        model = AlexNet()
        train(model, train_ds, validation_ds)
        test(model, test_ds)


if __name__ == "__main__":
    main()
