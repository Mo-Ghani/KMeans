# tensorflow dataset
from tensorflow.keras.datasets import mnist
# numpy
import numpy as np
# KMeans.py
from test2 import KMeans


def createMNIST(img_limit=60000):
    """
    Load MNIST data and transform image shapes
    :param img_limit: number of training images to return
    :return: training images and labels, test images and labels
    """

    if img_limit > 60000:
        print("Number of requested images to return is too large")
        print("Setting number of images to return to 60000 \n")
        img_limit = 60000

    (img_train, lab_train), (img_test, lab_test) = mnist.load_data()
    img_train = list(img_train)
    lab_train = list(lab_train)
    img_test = list(img_test)
    lab_test = list(lab_test)

    for i in range(len(img_train)):
        x = np.reshape(img_train[i], (1,784))
        img_train[i] = x[0]
    for i in range(len(img_test)):
        x = np.reshape(img_test[i], (1,784))
        img_test[i] = x[0]

    img_train = img_train[:img_limit]
    lab_train = lab_train[:img_limit]

    return (img_train, lab_train), (img_test, lab_test)


(img_train, lab_train), (img_test, lab_test) = createMNIST(5000)
model = KMeans(k=55)
model.centroidFinder(img_train,method="++")
model.training(img_train, lab_train)
model.evaluate(img_test, lab_test)
model.plotCentres()

