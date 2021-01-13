# numpy
import numpy as np
# plotting
import matplotlib.pyplot as plt
# standard libraries
from time import time
import random
from copy import deepcopy


class KMeans:

    """
    Class that assigns clusters to a data set using k-means, classifies each cluster, and evaluates
    the model performance on a test set
    """

    def __init__(self, k):
        """
        Creates a KMeans++ model with k clusters
        :param k: Predicted number of clusters
        """
        if type(k) is not int:
            print("k must be an integer")
            k = 1
        self.k = k
        self.clust_centres = None  # initialise list of cluster centroids
        self.clust_labels = None   # initialise list of cluster labels
        self.trained = False       # State of whether the model has trained
        print("KMeans model initialised with {} clusters \n".format(k))

    @staticmethod
    def distance(p1, p2):
        """
        Finds the distance between any two points, each defined by a vector of length N
        :param p1: Vector position of Point 1
        :param p2: Vector position of Point 2
        :return: Distance between Point 1 and 2
        """
        return np.linalg.norm(p1-p2)

    @staticmethod
    def cluster(centroids, data):
        """
        Given a data set and a list of centroids, this method labels each data point by finding it's nearest centroid
        :param centroids: List of centroids (centres of each cluster in a data set)
        :param data: List of data points
        :return: List containing the label for each data point
        """
        clusters = [np.argmin([KMeans.distance(point, cent) for cent in centroids]) for point in data]
        return clusters

    def centroidFinder(self, data_in, method="random"):
        """
        Initialises centroids in a data set
        :param data_in: Data with potential clusters
        :param method: algorithm for initialising clusters. Use "random" for random selection.
                       KMeans++ will be used for any other input.
        :return: List of potential centroids in the data.
        """

        t1 = time()
        data = deepcopy(data_in)
        # choose random datapoint to be the first centroid
        cents = [data[random.randint(0, len(data)-1)]]
        if method == "random":
            # choose k - 1 random points
            while len(cents) < self.k:
                cents.append(data[np.random.choice(range(len(data)), 1)[0]])
        else:
            # choose k-1 points according to kmeans++ algorithm
            while len(cents) < self.k:
                # find distance from every point to the nearest centroid
                dist = np.array([min([self.distance(point, c) for c in cents])**2 for point in data])
                # normalise these distances to get a distribution
                distnorm = dist/np.sum(dist)
                # randomly choose a new centroid using this distribution
                cents.append(data[np.random.choice(range(len(data)), 1, p=distnorm)[0]])
        self.clust_centres = cents
        t2 = time()
        print("Cluster centriods initialised after {} seconds \n".format(t2-t1))

    def training(self, data, labels):
        """
        Attempts to converge centroids to true cluster centre locations
        :param data: Data set with known classifications
        :param labels: Classifications of each data point
        :return: List of centroids of clusters in data, classification labels for each centroid
        """
        if len(labels) != len(data):
            print("Training data and labels should have the same length")
            return 0
        if self.clust_centres is None:
            # initialise cluster centers
            self.centroidFinder(self, data)
        cents = self.clust_centres
        k = self.k
        n = len(data)
        dims = len(data[0])
        print("Training with {} data points".format(len(data)))
        t1 = time()
        # training loop
        while True:
            cents_old = deepcopy(cents)
            clusters = self.cluster(cents, data)
            for i in range(k):
                # find points within the i'th cluster
                points = np.array([data[j] for j in range(n) if clusters[j] == i])
                cents[i] = np.mean(points, axis=0)
            error = np.zeros(dims)
            for c in range(len(cents)):
                error = np.add(error, np.subtract(cents_old[c], cents[c]))
            if all(e == 0 for e in error) is True:
                # break when all centroids do not change
                break
        t2 = time()
        # classify each centroid by finding the most common class in the points within it's cluster
        classify = []
        for i in range(len(cents)):
            labs = np.array([labels[j] for j in range(n) if clusters[j] == i])
            classify.append(np.argmax(np.bincount(labs)))
        self.clust_labels = classify
        self.trained = True
        print("Trained in {} seconds \n".format(t2-t1))

    def predict(self, predict):
        """
        Attempt to classify a list of unlabelled data points
        :param predict: List of unknown data points, to be classified
        :return: predictions of the label of each point in 'predict', estimation of certainty of each prediction
        """
        if not self.trained:
            print("Model not trained")
            return 0
        print("Predicting labels for {} data points".format(len(predict)))

        cents = self.clust_centres
        classes = self.clust_labels
        predictions = []
        t1 = time()
        for pred in predict:
            distances = np.array([self.distance(cent, pred) for cent in cents])
            mindist = np.argmin(distances)
            predictions.append(classes[int(mindist)])
        t2 = time()
        print("Predicted in {} seconds ({} seconds per image) \n".format(t2-t1, (t2-t1)/len(predict)))
        return predictions

    def evaluate(self, predict, labels):
        """
        Prints accuracy of model on a test set
        :param predict: Images to test on
        :param labels: Labels of test images
        """
        predictions = self.predict(predict)
        score = 0
        for i in range(len(predictions)):
            if predictions[i] == labels[i]:
                score += 1
        print("Accuracy: {}".format(score/len(predict)))

    def plotCentres(self, ncols=5, imgdims=(28,28)):
        """
        Plot array of images of the cluster centres
        :param ncols: Number of columns of images to plot (number of rows is inferred)
        :param imgdims: dimensions of each image to be plotted
        """
        centroids = self.clust_centres
        if self.clust_labels is not None:
            labels = self.clust_labels
        nrows = int(self.k-0.5)//ncols + 1

        # plot images in array
        fig, axs = plt.subplots(nrows, ncols)
        if self.clust_labels is not None:
            if nrows == 1:
                # plot row with labels as titles
                for i in range(self.k):
                    axs[i%ncols].imshow(np.reshape(centroids[i], imgdims))
                    axs[i%ncols].axis("off")
                    axs[i%ncols].set_title(str(labels[i]))
            else:
                # plot array with labels as titles
                for i in range(self.k):
                    axs[i//ncols, i%ncols].imshow(np.reshape(centroids[i], imgdims))
                    axs[i//ncols, i%ncols].axis("off")
                    axs[i//ncols, i%ncols].set_title(str(labels[i]))
        else:
            if nrows == 1:
                # plot row
                for i in range(self.k):
                    axs[i//ncols, i%ncols].imshow(np.reshape(centroids[i], imgdims))
                    axs[i//ncols, i%ncols].axis("off")
            else:
                # plot array
                for i in range(self.k):
                    axs[i//ncols, i%ncols].imshow(np.reshape(centroids[i], imgdims))
                    axs[i//ncols, i%ncols].axis("off")

        # remove axis on subplotsa that aren't used
        if self.k%ncols != 0:
            for i in range(ncols-self.k%ncols):
                axs[nrows-1, ncols-1-i].axis("off")
        plt.show()



