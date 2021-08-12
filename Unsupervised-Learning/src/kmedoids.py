# Libraries
import numpy as np
from preprocess_un import preprocess_num


class KMedoidsPAM:
    # Constructor for the KMedoids of Partition Around Medoids
    def __init__(self, dataset, k):
        # Get the preprocessed dataset with numerical features
        self.dataset = preprocess_num(dataset)

        # Set the k based on the argument
        self.k = k

        # Automatically fit and predict the dataset
        self.fit_predict()

    # Use Euclidean Distance as the distance metric
    def compute_distance(self, data_1, data_2):
        # Manual Approach of Euclidean Distance
        # distance = 0.0
        # for i in range(len(data_1) - 1):
        #     distance += (data_2[i] - data_1[i]) ** 2
        # return distance ** 0.5

        # The dataset will not contain any categorical features and labels
        # Use the numpy norm utils
        return np.linalg.norm(data_1 - data_2)

    # Randomly initialize k number of medoids
    def init_medoids(self):
        # Get random k indices in the dataset
        random_medoids = np.random.randint(len(self.dataset), size=self.k)

        # Initialize Numpy array to contain all medoids
        # Append all random medoids to the medoids
        medoids = np.array([self.dataset[medoid] for medoid in random_medoids])

        return medoids

    # Get closest medoid index from a data
    def get_closest_medoids(self, data, medoids):
        # Get all distances from the data to the medoids
        all_distances = np.array(
            [self.compute_distance(data, medoid) for medoid in medoids]
        )

        # Get the index of the closest medoid
        closest_medoid = np.argmin(all_distances)

        return closest_medoid

    # Create clusters and asisgn all data to the clusters
    def assign_clusters(self, medoids):
        # Initialize empty clusters
        clusters = [[] for _ in range(self.k)]

        # Iterate over all data
        for i in range(len(self.dataset)):
            # Get the closest medoids
            closest_medoid = self.get_closest_medoids(self.dataset[i], medoids)
            # Append the index of the data to the cluster of the closest medoid
            clusters[closest_medoid].append(i)

        return clusters

    # Compute total cost of every data to their medoids
    # The total cost is the Euclidean Distance of both points
    def compute_total_cost(self, clusters, medoids):
        # Initialize the total cost
        total_cost = 0.0

        # Iterate over all clusters
        for i in range(len(clusters)):
            # Iterate over all data in the cluster
            for data in clusters[i]:
                # Add all costs of the data and the medoid in the cluster
                total_cost += self.compute_distance(self.dataset[data], medoids[i])

        return total_cost

    # Get all non-medoids data
    def get_non_medoids(self):
        # Initialize empty list to contain all medoids
        non_medoids = []
        # Iterate over all data
        for data in self.dataset:
            # Check if the data is a medoid
            if data not in self.medoids:
                # Append the data if it is not a medoid
                non_medoids.append(data)

        return non_medoids

    # Check if the algorithm has reached convergence
    def is_convergence(self, current_cost, new_cost):
        return new_cost >= current_cost

    # Update Medoids
    def update_medoids(self):
        # Iterate until convergence occurs
        while True:
            # Get current medoids and current cost
            current_medoids = self.medoids
            current_cost = self.total_cost

            # Iterate over all medoids
            for medoid in self.medoids:
                # Iterate over all non-medoids data
                non_medoids = self.get_non_medoids()
                for non_medoid in non_medoids:
                    # Copy the current medoids
                    medoids_copy = np.copy(self.medoids)
                    # Change the medoid to the current non-medoid data
                    medoids_copy[medoid == self.medoids] = non_medoid
                    # Create different clusters with the new changed medoids
                    new_clusters = self.assign_clusters(medoids_copy)
                    # Calculate the total cost of the new assignment
                    new_total_cost = self.compute_total_cost(new_clusters, medoids_copy)
                    # Check if the cost is actually lower than the previous cost
                    if new_total_cost < current_cost:
                        # Change the medoids to the new copied one
                        current_medoids = medoids_copy
                        # Change the total cost
                        current_cost = new_total_cost

            # Check convergence of the algorithm
            if not self.is_convergence(self.total_cost, current_cost):
                # If not convergence, the medoids can be better
                # Update the new medoids and cost
                self.total_cost = current_cost
                self.medoids = current_medoids
            else:
                # If convergence, then the medoids is good enough
                return

    # Fit and predict the dataset
    def fit_predict(self):
        # Init all the medoids
        self.medoids = self.init_medoids()

        # Create and assign data to clusters
        self.clusters = self.assign_clusters(self.medoids)

        # Track current cost
        self.total_cost = self.compute_total_cost(self.clusters, self.medoids)

        # Update the medoids until convergence occurs
        self.update_medoids()

        # Assign the data to the best medoids
        self.assign_clusters(self.medoids)

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

    # Get all medoids
    def get_all_medoids(self):
        for i in range(len(self.medoids)):
            print("Cluster %d, medoids: " % (i + 1), self.medoids[i])

    # Predict the cluster of a new data
    def cluster(self, input_data):
        # Get all distances between the data and all medoids
        all_distances = [
            self.compute_distance(input_data, self.medoids[i])
            for i in range(len(self.medoids))
        ]

        # Get the index of the nearest centroid of a cluster
        nearest_medoids = all_distances.index(min(all_distances))

        # Return the cluster
        return nearest_medoids

    # Output the cluster
    def predict_cluster(self):
        # Output all medoids from the fitting
        self.get_all_medoids()

        # Get random input data
        input_data = self.get_random_data()
        print("Input Data:")
        print(input_data)

        # Output the predicted cluster
        print("Cluster of the input data: %d" % (self.cluster(input_data) + 1))
