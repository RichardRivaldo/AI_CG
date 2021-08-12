# Libraries
from numpy.lib.function_base import copy
from preprocess import preprocess_cat
import numpy as np


class IterativeDichotomiser3:
    # Constructor of Iterative Dichotomiser 3
    def __init__(self, dataset):
        # Full dataframe of the dataset
        dataframe = preprocess_cat(dataset)
        self.train_dataset = dataframe.iloc[: len(dataframe) - 100, :]
        self.test_dataset = dataframe.iloc[len(dataframe) - 100 :, :]

        # Automatically fit and train the model with the dataset
        self.fit_and_train(self.get_X_train(), self.get_y_train())

    # Get features of the Training Datasets
    def get_X_train(self):
        return self.train_dataset.iloc[:, :-1]

    # Get the label of the Training Dataset
    def get_y_train(self):
        return self.train_dataset.iloc[:, -1]

    # Get features of the Test Datasets
    def get_X_test(self):
        return self.test_dataset.iloc[:, :-1]

    # Get the label of the Test Dataset
    def get_y_test(self):
        return self.train_dataset.iloc[:, -1]

    # Calculate entropy of a column
    def entropy(self, data_col):
        # Get unique values in the column and their count
        unique_val, counts = np.unique(data_col, return_counts=True)

        # Initialize empty list to contain entropy for each unique values
        entropy_unique = []

        # Calculate entropy for every unique values
        for i in range(len(unique_val)):
            # Find the probability of the unique value is True on the whole column
            # Get the corresponding count from the counts list
            # The probability is the ratio of the count and all the total count
            probability = counts[i] / np.sum(counts)

            # Calculate the entropy with the log2 base
            entropy = -probability * np.log2(probability)

            # Append the entropy to the list
            entropy_unique.append(entropy)

        # Return the sum of every entropy in the column
        return np.sum(entropy_unique)

    # Calculate the Information Gain
    def info_gain(self, data, feature_name, target_name):
        # Find the total entropy of the target
        total_entropy = self.entropy(data[target_name])

        # Get unique values in the feature column and their count
        unique_val, counts = np.unique(data[feature_name], return_counts=True)

        # Initialized an empty list to contain the weighted entropy of the subset
        weighted_entropy = []

        # Calculate the weighted entropy with similar steps as above
        for i in range(len(unique_val)):
            # Get the probability of the subset
            subset_prob = counts[i] / np.sum(counts)

            # Map the subset data and drop the NaN values
            subset = data.where(data[feature_name] == unique_val[i]).dropna()
            subset = subset[target_name]

            # Find the entropy of the subset
            subset_entropy = self.entropy(subset)

            # Append the entropy to the weighted list
            weighted_entropy.append(subset_entropy)

        # Calculate the total weighted entropy
        total_weighted_entropy = np.sum(weighted_entropy)

        # Return the information gain, that is the difference of total entropy and total weighted entropy
        return total_entropy - total_weighted_entropy

    # Find the feature with the highest information gain
    def find_highest_gain(self, data, feature_name, target_name):
        # Get all information gain
        all_info_gain = [
            self.info_gain(data, feature, target_name) for feature in feature_name
        ]

        # Find the feature with the highest information gain to split on
        best_split = np.argmax(all_info_gain)

        # Return the feature name
        return feature_name[best_split]

    # Build the ID3 Decision Tree
    def build_decision_tree(
        self, original_data, copy_data, feature_name, target_name, parent_class=None
    ):
        # Base of the recursive

        # The Decision Tree is built based on Information Gain and Entropy
        # If the data is pure, it means that the data is homogenous
        # Automatically return the majority class
        unique_classes = np.unique(copy_data[target_name])
        if len(unique_classes) <= 1:
            return unique_classes[0]
        # The subset has no samples left
        # Automatically return majority class from the original data
        # Use np.argmax to find the index
        elif len(copy_data) == 0:
            target_col = original_data[target_name]
            unique, counts = np.unique(target_col, return_counts=True)
            major_class_idx = np.argmax(counts)
            return np.unique(target_col)[major_class_idx]
        # No feature in the dataset
        # Return the parent node's class
        elif len(feature_name) == 0:
            return parent_class

        # The recursions part
        else:
            # Find the parent node class of the current branch
            target_col = copy_data[target_name]
            major_class_idx = np.argmax(np.unique(target_col, return_counts=True)[1])

            # Set the parent node class
            parent_class = unique_classes[major_class_idx]

            # Get the best feature to split on
            best_feature = self.find_highest_gain(copy_data, feature_name, target_name)

            # Initialize an empty dict to represent the tree
            # The chosen feature to split on will be the parent node
            dict_tree = {best_feature: {}}

            # Pop the chosen feature from the list of feature names
            feature_name = [
                feature for feature in feature_name if feature != best_feature
            ]

            # Get all unique values of the parent node feature
            parent_value = np.unique(copy_data[best_feature])

            # Append child nodes to the parent node
            for value in parent_value:
                # Get the subset of the data and drop missing values
                data_subset = copy_data.where(copy_data[best_feature] == value).dropna()

                # Recursively create another tree as the child nodes
                rec_tree = self.build_decision_tree(
                    data_subset, original_data, feature_name, target_name, parent_class
                )

                # Append the child nodes to the parent node
                dict_tree[best_feature][value] = rec_tree

            # Return the tree
            return dict_tree

    # Fit and train the model
    def fit_and_train(self, features, target):
        # Copy the features and append the label to the feature
        data = features.copy()
        data[target.name] = target

        # Create the tree
        self.id3_tree = self.build_decision_tree(
            data, data, features.columns, target.name
        )

    # Filter the dataframe to get predicted data index
    def get_data_index(self, data_index):
        return self.test_dataset.iloc[[data_index], :]

    # Make a prediction
    def predict(self, data_index, decision_tree):
        # Get the dataframe of the predicted data
        predicted_data = self.get_data_index(data_index)

        # Convert the data into dict to suit the tree structure
        conv_predicted_data = predicted_data.to_dict(orient="records")

        # The method above will change the data from DataFrame to a List of dict
        # Traverse to all `records` of the data inside the list
        for pred_dict in conv_predicted_data:
            # Check all features in the predicted data
            for feature in list(pred_dict.keys()):
                # Check if the feature is in the tree feature
                if feature in list(decision_tree.keys()):
                    # Try getting deeper into the tree with the value of the feature in the predicted data
                    try:
                        result = decision_tree[feature][pred_dict[feature]]
                    except:
                        # Return default value of 1 if it is not possible
                        return 1

                    # Recursively iterate until there is no dict left to traverse
                    # Check if the result is still a dict, meaning that we can traverse it more
                    if isinstance(result, dict):
                        return self.predict(data_index, result)
                    else:
                        # The result is already the type of the label
                        # No more dict to traverse
                        return result

    # Output the prediction
    def output_prediction(self, data_index):
        # Output the predicted data:
        print("Input Data:")
        print(self.get_X_test().iloc[data_index, :].to_numpy())

        # Getting the prediction
        prediction = int(self.predict(data_index, self.id3_tree))

        # Output the expected predicted churn value
        print("Expected Churn: %d" % (self.get_y_test().iloc[data_index]))
        print("Predicted Churn: %d" % prediction)
