# K-MEANS CLUSTERING
Mo Ghani : https://github.com/Mo-Ghani/KMeans
#########################################################################################

## BACKGROUND
The Python program "KMeans.py" provides a tool to cluster a given data set with N
datapoints, each with M features, into some integer k clusters. This is done via the 
method of k-means clustering. As an example, I have used "mnist.py" to create a digit
recognition model, which is trained using the MNIST data set (which can be found at 
http://yann.lecun.com/exdb/mnist/).


## PYTHON VERSION AND REQUIRED PACKAGES
Python v3.7
KMeans.py - NumPy v1.18.1, MatPlotLib v3.1.3, standard libraries(time, random, copy)
mnist.py - Numpy v1.18.1, tensorflow v2.1.0

Here tensorflow is only used to easily obtain the MNIST data, and is not used to create
any models.

## INSTRUCTIONS - KMeans.py
This program contains all the machinery for performing the clustering.

Initialising a model:

On initialisation, the model requires only the expected number of clusters k to be given.

Initialising centroids:

The method "centroidFinder" is called to initialise the cluster centroids. This function
takes an N x M matrix which contains N datapoints with M features, and an optional
initialisation method parameter. The default is "random", which will choose random
datapoints as the cluster centres. setting the method to "++" will use the KMeans++
algorithm, which randomly chooses new centroids from a weighted probability distribution
based on the square of the distance to the nearest centroids.

Training the model:

The method "training" is called to train the model. This function takes an N x M matrix
which contains N datapoints with M features, and an N-vector which contains the 
corresponding labels for each datapoint. The function then attempts to converge the
centroids to their optimal positions.

Predicting labels:

Once the model is trained, The method "predict" is called to to predict the label of
unknown datapoints. The function takes an N x M matrix which contains N datapoints with
M features, and returns a list of labels assigned to each datapoint in the
input matrix.

Evaluating the model:

The method "evaluate" is called to find the accuracy of the model on a test set. The
function takes an N x M matrix which contains N datapoints with M features, and an
N-vector which contains the corresponding labels for each datapoint in the input matrix.
The function will print the model's accuracy as the ratio of the correctly predicted 
labels and N.

Plotting the centroids:

The method "plotCentres" is called to plot an array containing heatmaps of every centroid. 
The function only has optional parameters, and takes the number of columns of the heatmap
array (number of rows is inferred), and the dimensions of a single heatmap to be plotted.


## INSTRUCTIONS - mnist.py
This program contains an example usage of KMeans.py to classify digits from the MNIST
data set. 

Loading the data:
The function "createMNIST" is called to download the MNIST data (if not already
downloaded using tensorflow) and transform the data into the correct format. This function
only has optional parameters, and takes the desired number of training images to be
returned. 

I found that using around 50 clusters and 60000 images produced the greatest accuracy,
however this took around 45 minutes to train on my machine. The resulting accuracy was
0.856

For models that can train faster, I find a setting of 55 clusters trained on 5000 images
gives an accuracy of 0.798, and only takes 2 minutes to train on my machine. 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
