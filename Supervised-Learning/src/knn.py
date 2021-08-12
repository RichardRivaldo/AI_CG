# Libraries
from preprocess import preprocess_num
from collections import Counter
import numpy as np


class KNNClassifier:
    # Constructor of KNN Classifier
    def __init__(self, dataset, k_nearest):
        # K Constant
        self.k_nearest = k_nearest
        # Array of vectors of each data to ease calculations
        arr_dataset = preprocess_num(dataset).to_numpy()
        self.train_dataset = arr_dataset[: len(arr_dataset) - 100]
        self.test_dataset = arr_dataset[len(arr_dataset) - 100 :]

    # Euclidean Distance
    def compute_distance(self, data_1, data_2):
        # Manual Approach of Euclidean Distance
        # distance = 0.0
        # for i in range(len(data_1) - 1):
        #     distance += (data_2[i] - data_1[i]) ** 2
        # return distance ** 0.5

        # Using Numpy Library
        # Ignore the last column as it is the label
        return np.linalg.norm(data_1[:-1] - data_2[:-1])

    # Finding K-Nearest Neighbors
    # input_data is the predicted data
    def k_nearest_neighbors(self, input_data):
        # Initialize empty array to contain all distances calculation
        all_distance = []

        # Iterate over all data in the dataset and calculate the distance
        # Append the calculation and the corresponding data to the list
        for train_data in self.train_dataset:
            data_and_dist = (self.compute_distance(input_data, train_data), train_data)
            all_distance.append(data_and_dist)

        # Sort the array with the distance as the key
        all_distance.sort(key=lambda x: x[0])

        # Reverse and slice K Nearest Neighbors
        k_nearest = all_distance[: self.k_nearest]

        return k_nearest

    # Get nearest neighbors categories
    def get_categories(self, k_nearest):
        # The element at index 1 of each neighbor is the vector of the full data
        # The last element will be the label of the categories of each neighbor
        categories = [int(neighbor[1][-1]) for neighbor in k_nearest]
        return categories

    # Predict and Classify
    def predict(self, input_data):
        # Get the neighbors of the input data from the training set
        k_nearest_neighbors = self.k_nearest_neighbors(input_data)

        # Extract all of the neighbors category
        # Use Counter library to do the task
        categories = Counter(self.get_categories(k_nearest_neighbors))

        # Find best category for the input data
        # Use max function and set the key to dict key since Counter is a dict
        best_category = max(categories, key=categories.get)

        return best_category

    # Output the prediction
    def output_prediction(self, input_data):
        # Output the predicted data:
        print("Input Data:")
        print(input_data[:-1])

        # Get prediction
        prediction = self.predict(input_data)

        # Output the expected predicted churn value
        print("Expected Churn: %d" % (input_data[-1]))
        print("Predicted Churn: %d" % prediction)
