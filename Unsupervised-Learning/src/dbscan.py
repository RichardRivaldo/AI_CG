# Libraries
import numpy as np
from queue import Queue
from preprocess_un import get_clean_dataset


class DBScan:
    # Constructor of the Density-Based Spatial Clustering of Applications with Noise
    def __init__(self, dataset, epsilon, min_pts):
        # Read the whole dataset
        self.dataset = get_clean_dataset(dataset)

        # Add the label of -1 to indicate that the data is unclustered
        self.dataset = np.append(
            self.dataset, np.full((len(self.dataset), 1), -1), axis=1
        )

        # Assign both epsilon and minimum points to the clustering model
        self.epsilon = epsilon
        self.min_pts = min_pts

        # Initialize global clusters label and noise label to track the label
        self.cluster_label = 0
        self.noise_label = 0

        # Automatically fit the dataset
        self.fit_predict()

    # Euclidean Distance (and Hamming Distance indirectly)
    def compute_distance(self, data_1, data_2):
        # Manual Approach of Euclidean Distance
        # distance = 0.0
        # for i in range(len(data_1) - 1):
        #     distance += (data_2[i] - data_1[i]) ** 2
        # return distance ** 0.5

        # The only categorical feature in the dataset is the gender, mapped into 0 and 1
        # Check if the gender of both data is same
        if data_1[0] == data_2[0]:
            # Just return the rest of euclidean distance
            # Using Numpy Library
            # Ignore the last column as it is the artificial label
            return np.linalg.norm(data_1[1:-1] - data_2[1:-1])
        else:
            # The gender is not identical
            # The Hamming Distance will be 1 + euclidean distance of the rest
            return 1 + np.linalg.norm(data_1[1:-1] - data_2[1:-1])

    # Get all neighbors from a data
    def get_all_neighbors(self, data):
        # Initialize an empty list to contain all neighbors
        neighbors = []

        # Iterate over the whole data
        for i in range(len(self.dataset)):
            if self.compute_distance(data, self.dataset[i]) <= self.epsilon:
                # Append the neighboring data index if the distance is less or equal than epsilon
                neighbors.append(i)

        return neighbors

    # Fit and predict the dataset to the model
    def fit_predict(self):
        # Initialize empty cluster
        clusters = None

        # Iterate over all data in the dataset
        for i in range(len(self.dataset)):
            # Check if the data is already labeled
            if self.dataset[i][-1] != -1:
                # Just skip the data
                continue

            # Get all neighbors of the data
            neighbors = self.get_all_neighbors(self.dataset[i])

            # Check the number of points in the neighborhood
            if len(neighbors) < self.min_pts:
                # If the number of neighbors is less than the number of minimum points
                # Label the data as a noise and skip the data
                self.dataset[i][-1] = self.noise_label
                continue

            # If not, then a cluster can be made
            # Increment the cluster label and assign it to the cluster
            self.cluster_label += 1
            self.dataset[i][-1] = self.cluster_label

            # Initialized a seed to contain all neighborhoods or clusters found
            clusters = neighbors

            # Create a queue to check and expand each neighbor further
            # Add all original neighbors
            queue_neighbors = Queue()
            for neighbor in neighbors:
                queue_neighbors.put(neighbor)

            # Expand all neighbors until the queue is empty
            while not queue_neighbors.empty():
                # Get current expanded neighbors
                current_neighbor = queue_neighbors.get()

                # Check current label of the current neighbor
                # If the label is noise, change it to the current cluster label
                if self.dataset[current_neighbor][-1] == 0:
                    self.dataset[current_neighbor][-1] = self.cluster_label
                # If the label is not unclassified, meaning it is already clustered, skip the data
                if self.dataset[current_neighbor][-1] != -1:
                    continue
                # Else, the neighbor can be labeled to the current cluster label
                self.dataset[current_neighbor][-1] = self.cluster_label

                # Get all neighbors of the current neighbor
                neighbors_of_current = self.get_all_neighbors(
                    self.dataset[current_neighbor]
                )

                # Add the neighbors if it is eligible
                if len(neighbors_of_current) >= self.min_pts:
                    for neighbor in neighbors_of_current:
                        # Check if the neighbor is already in the clusters or not
                        if neighbor not in clusters:
                            # Append the neighbor to the clusters and queue
                            clusters.append(neighbor)
                            queue_neighbors.put(neighbor)

    # Create a random data
    def get_random_data(self):
        # Get max and min age, annual income, and spending score
        max_age, max_income, max_spending = np.amax(self.dataset[:, 1:-1], axis=0)
        min_age, min_income, min_spending = np.amin(self.dataset[:, 1:-1], axis=0)

        # Return between 0 and 1 for gender
        random_gender = np.random.randint(2)
        # Return random age between max and min age
        random_age = np.random.randint(min_age, max_age + 1)
        # Return random income between max and min income
        random_income = np.random.randint(min_income, max_income + 1)
        # Return random spending between max and min spending
        random_spending = np.random.randint(min_spending, max_spending + 1)

        return np.array((random_gender, random_age, random_income, random_spending, -1))

    # Cluster a new data
    def cluster(self, input_data):
        # Get all neighbors of the input data
        input_neighbors = self.get_all_neighbors(input_data)

        # If the input data has no neighbors or less than minimum points, label it as noise
        if len(input_neighbors) < self.min_pts:
            return self.noise_label

        # Get all distances of the neighbors
        neighbor_distances = np.array(
            [
                self.compute_distance(input_data, self.dataset[neighbors])
                for neighbors in input_neighbors
            ]
        )

        # Zip, combine, and sort both list based on ascending distances
        neighbor_info = list(zip(input_neighbors, neighbor_distances))
        neighbor_info.sort(key=lambda x: x[1])

        # Iterate over the closest neighbor first
        for neighbors in neighbor_info:
            # Check if the neighbor is a core point
            neighbor_array = self.dataset[neighbors[0]]
            if len(self.get_all_neighbors(neighbor_array)) >= self.min_pts:
                # If yes, automatically cluster them together
                return neighbor_array[-1]

        # Default return the label of the closest point if no core points found
        return self.dataset[neighbors[0]][-1]

    # Get all clusters
    def get_all_clusters(self):
        # Get all clusters and counts of the model
        clusters, counts = np.unique(self.dataset[:, -1], return_counts=True)

        # Iterate over both list
        for i in range(len(clusters)):
            print("Cluster %d: %d data " % (clusters[i], counts[i]))
        print()

    # Predict the cluster of an input data
    def predict_cluster(self):
        # Show all clusters
        self.get_all_clusters()

        # Get random input data
        input_data = self.get_random_data()
        print("Input Data: ")
        print(input_data[:-1])

        # Output the predicted label
        print("Cluster of the input data: %d" % self.cluster(input_data))
