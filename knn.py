import pandas as pd
import numpy as np


# Splits the dataset into training and testing data
# Returns a tuple containing the test data (at index 0) and the training data
def split_data(dataset: pd.DataFrame, test_split: float, validation_split: float) -> tuple:
    test_split_index = int(len(dataset) * test_split)
    validation_split_index = int(len(dataset) * (test_split + validation_split))

    # Splitting the dataset into test, validation and training data
    test_data = dataset[:test_split_index]
    validation_data = dataset[test_split_index:validation_split_index + 1]
    training_data = dataset[validation_split_index + 1:]

    return (test_data, validation_data, training_data)


# Finds the euclidean distance between two rows in a Pandas DataFrame.
# It treats the values in each column of the rows as a point in a dimension in space.
#
# NOTE: This function assumes all of the values of a row are numeric and treats the 
# rows like vectors in order to find the distance.
def euclidean_distance(r1: pd.Series, r2: pd.Series):
    v1 = np.array(r1)
    v2 = np.array(r2)

    return  np.linalg.norm(v1 - v2)


# This class will represent an instance of the KNN model with a static K
class KNN:


    def __init__(self, K=1, dist_fn=euclidean_distance):
        self.dist_fn = dist_fn
        self.K = K


    # Memorizes the data
    def fit(self, training_data: pd.DataFrame) -> None:
        self.training_data = training_data


    # Predicts the labels of samples in the test dataset and returns a percentage accuracy
    def predict(self, test_data: pd.DataFrame) -> int:
        correct_predictions = 0

        # Looping through each point in the test dataset
        for _, r1 in test_data.iterrows():

            # Dictionary to hold the distances to each other point
            neighbors = {}

            # Getting the distance between current test point and all training points
            for i, r2 in self.training_data.iterrows():

                # Taking the rows without ID and label to find the distance between the two
                dist = self.dist_fn(r1.iloc[1:-1], r2.iloc[1:-1])
                neighbors[i] = dist

            # Finding the k-nearest neighbors
            nearest_neighbors = sorted(neighbors.items(), key=lambda x: x[1])[:self.K]
            nearest_neighbors = [x[0] for x in  nearest_neighbors] # List containing only indexes of nearest neighbors

            # Getting the predicted label for the new point
            pred_label = self.get_weighted_label(nearest_neighbors)

            # Checking to see if predicted label is correct
            if pred_label == int(r1.iloc[-1]):
                correct_predictions += 1

        # Returning the accuracy of the model
        return correct_predictions / len(test_data)


    # Predicts the label of a point given a list of IDs of the nearest neighbors to that point
    # NOTE: Assumes labels are the last value in the row
    def get_weighted_label(self, nearest_neighbors: list):
        label_votes = {}

        # Counting the labels for each 
        for n in nearest_neighbors:
            label = int(self.training_data.iloc[n].iloc[-1])
            
            # Adding label vote to the dictionary
            if label_votes.__contains__(label):
                label_votes[label] = label_votes[label] + 1
            else:
                label_votes[label] = 1

        # Returning the label with the most votes (sorted in ascending order)
        return sorted(label_votes.items(), key=lambda x: x[1])[-1][0]

            
