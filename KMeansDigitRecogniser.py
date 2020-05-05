import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from copy import deepcopy
from collections import Counter
from keras.datasets import mnist
import itertools


class KMeans:

    """
    Class that applies the k-means++ algorithm to a data set, and classifies
    the data into the specified number of clusters.

    """

    def __init__(self, k):
        """
        Creates a KMeans++ model with k clusters
        :param k: Predicted number of clusters
        """
        self.k = k

    @staticmethod
    def distance(p1, p2):
        """
        Finds the distance between any two points defined by a vector of length N
        :param p1: Vector position of Point 1
        :param p2: Vector position of Point 2
        :return: Distance between Point 1 and 2
        """
        distance = 0
        for i in range(len(p1)):
            distance += (float(p1[i]) - float(p2[i]))**2
        return distance**0.5

    @staticmethod
    def cluster(centroids, data):
        """
        Given a data set and a list of centroids, this method labels each data point by finding it's nearest centroid
        :param centroids: List of centroids (centres of a cluster in a data set)
        :param data: List of data points
        :return: List containing the label for each data point
        """
        clusters = [np.argmin([KMeans.distance(point, cent) for cent in centroids]) for point in data]
        return clusters

    def centroidFinder(self, data1):
        """
        Initialises centroids in a data set
        :param data1: Data with potential clusters
        :return: List of potential centroids in the data.
        """
        data = deepcopy(data1)
        cent = [data[random.randint(0, len(data))-1]]
        while len(cent) < self.k:
            dist = [min([KMeans.distance(point, c) for c in cent])**2 for point in data]
            disttot = sum(dist)
            distNorm = [d/disttot for d in dist]
            ind = range(len(data))
            cent.append(data[np.random.choice(ind, 1, p= distNorm)[0]])
        return cent

    def training(self, data, labels):
        """
        Attempts to converge centroids to true cluster centre locations
        :param data: Data set with known classifications
        :param labels: Classifications of each data point
        :return: List of centroids of clusters in data, classification labels for each centroid
        """
        cents = KMeans.centroidFinder(self, data)
        while True:
            error = np.zeros(len(cents[0]))
            cents_old = deepcopy(cents)
            clusters = KMeans.cluster(cents, data)
            for i in range(len(cents)):
                points = np.array([data[j] for j in range(len(data)) if clusters[j] == i])
                for ind in range(len(cents[i])):
                    cents[i][ind] = np.mean(points[:, ind])
                error = np.add(error, np.subtract(cents_old[i], cents[i]))
            if all(e == 0 for e in error) is True:
                break
        classify = []
        for i in range(len(cents)):
            labs = np.array([labels[j] for j in range(len(labels)) if clusters[j] == i])
            classify.append(Counter(labs).most_common()[0][0])
        return cents, classify

    def predict(self, predict, data, labels):
        """
        Attempt to classify a list of unlabelled data points
        :param predict: List of unknown data points, to be classified
        :param data: List of known data points
        :param labels: Labels corresponding to known data points
        :return: predictions of the label of each point in 'predict', estimation of certainty of each prediction
        """
        cents, classes = KMeans.training(self, data, labels)
        predictions = []
        certainty = []
        for pred in predict:
            distances = np.array([KMeans.distance(cent, pred) for cent in cents])
            mindist = np.argmin(distances)
            maxdist = np.sum(distances)
            predictions.append(classes[int(mindist)])
            certainty.append((maxdist - distances[mindist]) / maxdist)
        return predictions, certainty


def datasorter(data, labels):
    """
    Sorts training data into groups of simillar labels
    :param data: data to be sorted
    :param labels: labels of each data point
    :return: list containing groups of data points with simillar labels, list containing label of each data points
    """
    arranged = [[] for s in range(10)]
    arlab = [[] for s in range(10)]
    for i in range(len(data)):
        ind = labels[i]
        arranged[ind].append(data[i])
        arlab[ind].append(labels[i])
    return arranged, arlab


def predictor(centers, labels, test):
    """
    predicts the label of each data point in parameter 'test'
    :param centers: centroids of a dataset, found using the method of KMeans
    :param labels: the labels of each centroid
    :param test: unknown data points, to be classified
    :return:
    """
    classes = []
    for t in test:
        dist = np.array([KMeans.distance(t, c) for c in centers])
        distMin = np.argmin(dist)
        classes.append(labels[distMin][0])
    return classes

# Load MNIST dataset, and transform shapes into a usable format

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

# Group training data points with the same labels

trainData, trainLab = datasorter(img_train, lab_train)
testData, testLab = img_test, lab_test

# Find centre of cluster for each group

centroids = []
model = KMeans(k=1)
for c in range(len(trainData)):
    centroids.append(model.training(trainData[c], trainLab[c])[0][0])


# For an unknown data point, assign the label of the closest centroid
# Make predictions of labels of known data points, and test if the model is correct.
# Find fraction of correct predictions

predictions = predictor(centroids, trainLab, testData)

score = 0
for i in range(len(predictions)):
    if predictions[i] == testLab[i]:
        score += 1
accuracy = score/len(testLab)

print(accuracy)

# Plot each centroid

fig, axs = plt.subplots(2, 5)
for i in range(10):
    if i < 5:
        axs[0, i].imshow(np.reshape(centroids[i], (28, 28)))
    else:
        axs[1, i-5].imshow(np.reshape(centroids[i], (28, 28)))
plt.show()

