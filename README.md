# KMeans-digit-Recognition
Recognition of digits in a 28x28 pixel image, using the KMeans++ algorithm and the MNIST digit dataset.

Automatically loads the MNIST dataset, provided all the packages are installed on the machine.

Running "KMeansDigitRecogniser" begins the learning process. The MNIST training dataset is loaded, sorted, and rearranged so that each
28x28 array becomes one 784 vector. 10 cluster centroids are found using a handmade KMeans++ algorithm. The testing set is then used
to validate the model, and the accuracy is printed.

