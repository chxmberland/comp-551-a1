import pandas as pd
import numpy as np

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


    def __init__(self, K = 1, dist_fn = euclidean_distance):
        self.dist_fn = dist_fn
        self.K = K


    # Memorizes the data
    def fit(self, training_features, training_target):
        self.x = training_features
        self.y = training_target


    # Predicts the labels of samples in the test dataset and returns a percentage accuracy
    def predict(self, test_data: pd.DataFrame) -> list:

        predictions = []

        # Looping through each point in the test dataset
        for i in range(0, len(test_data)):
            test_row = test_data.iloc[i]

            # Dictionary to hold the distances to each other point
            neighbors = {}

            # Getting the distance between current test point and all training points
            # j being the index of each training row
            for j in range(0, len(self.training_data)):
                train_row = self.training_data.iloc[j]

                # Taking the rows to find the distance between the two
                dist = self.dist_fn(test_row, train_row)
                neighbors[j] = dist

            # Finding the k-nearest neighbors
            nearest_neighbors = sorted(neighbors.items(), key=lambda x: x[1])[:self.K] # sorted list of tuples
            nearest_neighbors = [x[0] for x in  nearest_neighbors] # List containing only indexes of nearest neighbors

            # Getting the predicted label for the new point
            pred_label = self.get_weighted_label(nearest_neighbors)

            # Checking to see if predicted label is correct
            if pred_label == int(test_row.iloc[-1]):
                correct_predictions += 1

        #
        return predictions


    # Predicts the label of a point given a list of IDs of the nearest neighbors to that point
    # NOTE: Assumes labels are the last value in the row
    def get_weighted_label(self, nearest_neighbors: list):
        label_votes = {}

        # Counting the labels for each
        for i in nearest_neighbors:
            r = self.training_data.iloc[i]
            label = r.iloc[-1] # Label value
            
            # Adding label vote to the dictionary
            if label_votes.__contains__(label):
                label_votes[label] = label_votes[label] + 1
            else:
                label_votes[label] = 1

        # Returning the label with the most votes (sorted in ascending order)
        return sorted(label_votes.items(), key=lambda x: x[1])[-1][0]