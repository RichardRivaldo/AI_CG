# Libraries
from numpy.core.function_base import logspace
from preprocess import preprocess_num
import numpy as np


class LogisticRegression:
    # Constructor of Logistic Regression Classifier
    def __init__(self, dataset, lr, epochs):
        # Learning Rate
        self.lr = lr

        # Number of epochs
        self.epochs = epochs

        # Full dataframe of the dataset, unlike the one in KNN
        # Use the get_X or get_y to get the data in Numpy array
        dataframe = preprocess_num(dataset)
        self.train_dataset = dataframe.iloc[: len(dataframe) - 100, :]
        self.test_dataset = dataframe.iloc[len(dataframe) - 100 :, :]

        # Weight and intercept of the regression
        # The intercept is initialized as 0.0
        # Weight is initialized as zeros for all features
        # The dimension of the weight is the number of features - 1 as the label
        self.weight = np.zeros(self.train_dataset.shape[1] - 1)
        self.intercept = 0.0

        # Automatically fit and train the model to the dataset after constructed
        self.train_and_fit()

    # Normalize the data features to avoid overflow and zero division by Sigmoid later
    def normalize(self, dataframe):
        dataframe = dataframe.iloc[:, :-1]
        return (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())

    # Get features of the Training Datasets
    def get_X_train(self):
        X_train = self.normalize(self.train_dataset)
        return X_train.to_numpy()

    # Get the label of the Training Dataset
    def get_y_train(self):
        y = self.train_dataset.iloc[:, -1]
        return y.to_numpy()

    # Get features of the Test Datasets
    def get_X_test(self):
        X_test = self.normalize(self.test_dataset)
        return X_test.to_numpy()

    # Get the label of the Test Dataset
    def get_y_test(self):
        y = self.test_dataset.iloc[:, -1]
        return y.to_numpy()

    # Define sigmoid function
    def sigmoid_function(self, z):
        return 1 / (1 + np.exp(-z))

    # Apply weight to inputs with Dot Product
    def apply_weight(self, inputs):
        return np.dot(inputs, self.weight)

    # Calculate the activation function using on the sigmoid function
    def activation_function(self):
        return self.sigmoid_function(
            self.apply_weight(self.get_X_train()) + self.intercept
        )

    # Calculate cost function
    # Cost Function -> Calculated each epoch, using loss function which is calculated on each data
    def cost_function(self):
        # Get the activation value
        sigmoid_z = self.activation_function()
        # Data Count
        data_count = len(self.get_X_train())

        return (-1 / data_count) * np.sum(
            self.get_y_train() * np.log(sigmoid_z)
            + (1 - self.get_y_train()) * np.log(1 - sigmoid_z)
        )

    # Gradient Descent
    def gradient_descent(self):
        # Data Count
        data_count = len(self.get_X_train())

        # Get the activation value
        activation = self.activation_function()

        # Count dW and dB
        act_min_y = activation - self.get_y_train()
        dW = (1 / data_count) * np.dot(act_min_y.T, self.get_X_train())
        db = (1 / data_count) * np.sum(act_min_y)

        return (dW, db)

    # Update weights per epoch
    def update_weight(self):
        # Get the gradient
        dW, db = self.gradient_descent()

        # Apply gradient descent to the current hyperparameter based on the learning rate
        self.weight -= self.lr * dW
        self.intercept -= self.lr * db

    # Create a training and fitting function
    def train_and_fit(self):
        # Use the gradient descent to update the weights and the intercept
        print("Training Session...")
        for epoch in range(self.epochs):
            self.cost = self.cost_function()

            # Update the weight
            self.update_weight()

            # Output the cost
            print("Epoch %d, Cost: %f" % (epoch + 1, self.cost))
        print("Finished Learning!")

    # Predict and Classify
    def predict(self, data_idx):
        # Getting the data
        input_data = self.get_X_test()[data_idx]

        # Calculate the z score
        sigmoid_input = self.sigmoid_function(
            self.apply_weight(input_data) + self.intercept
        )

        return np.where(sigmoid_input > 0.5, 1, 0)

    # Output the prediction
    def output_prediction(self, data_idx):
        # Output the predicted data:
        print("Input Data:")
        print(self.get_X_test()[data_idx])

        # Get prediction
        prediction = self.predict(data_idx)

        # Output the expected predicted churn value
        print("Expected Churn: %d" % (self.get_y_test()[data_idx]))
        print("Predicted Churn: %d" % prediction)
