# Libraries
import numpy as np
from preprocess_un import preprocess_num


class KMeans:
    # Constructor of the KMeans Model
    def __init__(self, dataset, k):
        # Get the clean dataset to the model
        self.dataset = preprocess_num(dataset)

        # Set the k number of centroids
        self.k = k

        # Set the default number of maximum iteration and SSE tolerance value
        self.max_iter = 300
        self.tolerance = 0.01

        # Automatically fit and predict the dataset
        self.fit_predict()

    # Euclidean Distance
    def compute_distance(self, data_1, data_2):
        # Manual Approach of Euclidean Distance
        # distance = 0.0
        # for i in range(len(data_1) - 1):
        #     distance += (data_2[i] - data_1[i]) ** 2
        # return distance ** 0.5

        # The dataset will not contain any categorical features and labels
        # Use the numpy norm utils
        return np.linalg.norm(data_1 - data_2)

    # Randomly initialize k centroids
    def init_centroids(self):
        # Get random k indices in the dataset
        random_centroids = np.random.randint(len(self.dataset), size=self.k)

        # Initialize a dict to contain all centroids
        centroids = {}

        # Insert the data to the dict
        for i in range(self.k):
            centroids[i] = self.dataset[random_centroids[i]]

        return centroids

    # Initialize a dict that contains the clusters
    def init_clusters(self):
        # Initialize the dict
        clusters = {}

        # Add k number of clusters represented by empty list
        for i in range(self.k):
            clusters[i] = []

        return clusters

    # Assign a data to a nearest centroid
    def assign_data(self, data):
        # Get all distances between the data and all centroids
        all_distances = [
            self.compute_distance(data, self.centroids[centroid])
            for centroid in self.centroids
        ]

        # Get the index of the nearest centroid
        nearest_centroid = all_distances.index(min(all_distances))

        # Assign the data to the centroid
        self.clusters[nearest_centroid].append(data)

    # Update and move the centroids
    def update_centroids(self):
        # Iterate over every clusters
        for cluster in self.clusters:
            # Set the centroid to average value of each cluster it is in
            self.centroids[cluster] = np.average(self.clusters[cluster], axis=0)

    # Check convergence of each centroid
    def check_convergence(self, old_centroids):
        # Iterate over all centroids
        for centroid in self.centroids:
            # Get the old and updated centroid of the same cluster
            prev_centroid = old_centroids[centroid]
            curr_centroid = self.centroids[centroid]

            # Calculate the error between the two centroids
            error = np.sum((curr_centroid - prev_centroid) / prev_centroid * 100.0)

            # Check if the error is still greater than the tolerance value
            # If yes, then the algorithm hasn't reached convergence
            if error > self.tolerance:
                return False

        # The algorithm has reached the convergence
        return True

    # Fit and predict the dataset fed to the data
    def fit_predict(self):
        # Initialize the atribute of centroids randomly
        self.centroids = self.init_centroids()

        # Iterate until the number of maximum iteration or convergence reached
        for i in range(self.max_iter):
            # Initialize the clusters
            self.clusters = self.init_clusters()

            # Assign all data to the nearest centroid
            for data in self.dataset:
                self.assign_data(data)

            # Track previous centroids by copying them before updating all centroids
            old_centroids = dict(self.centroids)

            # Update all centroids
            self.update_centroids()

            # Check the convergence of the algorithm
            is_convergence = self.check_convergence(old_centroids)

            # If convergence is reached, stop the iteration
            if is_convergence:
                return

    # Create a random data
    def get_random_data(self):
        # Get max and min age, annual income, and spending score
        max_age, max_income, max_spending = np.amax(self.dataset, axis=0)
        min_age, min_income, min_spending = np.amin(self.dataset, axis=0)

        # Return random age between max and min age
        random_age = np.random.randint(min_age, max_age + 1)
        # Return random income between max and min income
        random_income = np.random.randint(min_income, max_income + 1)
        # Return random spending between max and min spending
        random_spending = np.random.randint(min_spending, max_spending + 1)

        return np.array((random_age, random_income, random_spending))

    # Get centroids and clusters
    def get_all_centroids(self):
        for cluster in self.clusters:
            print("Cluster %d, centroid: " % (cluster + 1), self.centroids[cluster])

    # Predict the cluster of a new data
    def cluster(self, input_data):
        # Get all distances between the data and all centroids
        all_distances = [
            self.compute_distance(input_data, self.centroids[centroid])
            for centroid in self.centroids
        ]

        # Get the index of the nearest centroid of a cluster
        nearest_centroid = all_distances.index(min(all_distances))

        # Return the cluster
        return nearest_centroid

    # Output the cluster
    def predict_cluster(self):
        # Output all centroids from fitting
        self.get_all_centroids()

        # Get random input data
        input_data = self.get_random_data()
        print("Input Data: ")
        print(input_data)

        # Output the predicted label
        print("Cluster of the input data: %d" % (self.cluster(input_data) + 1))
