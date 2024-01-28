import pandas as pd
import numpy as np


# Splits the dataset into training and testing data
# Returns a tuple containing the test data (at index 0) and the training data
def split_data(dataset: pd.DataFrame, test_percentage: float, validation_percentage: float, label_column: int) -> tuple:

    # Shuffling the rows of the dataframe
    shuffled_df = dataset.sample(frac=1).reset_index(drop=True)

    # Getting the index at which we need to split the incoming dataset
    test_split_index = int(len(shuffled_df) * test_percentage)
    validation_split_index = int(len(shuffled_df) * (test_percentage + validation_percentage))

    # Splitting the dataset into test, validation and training data
    test_data = shuffled_df[:test_split_index]
    validation_data = shuffled_df[test_split_index:validation_split_index + 1]
    training_data = shuffled_df[validation_split_index + 1:]

    # Extracting the labels
    test_x = test_data.drop(test_data.columns[label_column], axis=1)
    test_y = test_data.iloc[:, label_column].to_numpy()
    validation_x = validation_data.drop(validation_data.columns[label_column], axis=1)
    validation_y = validation_data.iloc[:, label_column].to_numpy()
    training_x = training_data.drop(training_data.columns[label_column], axis=1)
    training_y = training_data.iloc[:, label_column].to_numpy()

    return (test_x, test_y, validation_x, validation_y, training_x, training_y)


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
    def fit(self, training_data_x: pd.DataFrame, training_data_y: np.ndarray) -> None:
        self.training_data_x = training_data_x
        self.training_data_y = training_data_y


    # Predicts the labels of samples in the test dataset and returns a percentage accuracy
    def predict(self, input_x: pd.DataFrame, input_y: np.ndarray) -> int:
        correct_predictions = 0

        # Looping through each point in the test dataset
        for i in range(0, len(input_x)):
            r1 = input_x.iloc[i]

            # Dictionary to hold the distances to each other point
            neighbors = {}

            # Getting the distance between current test point and all training points
            for j in range(0, len(self.training_data_x)):
                r2 = self.training_data_x.iloc[j]

                # Taking the rows without ID and label to find the distance between the two
                dist = self.dist_fn(r1, r2)
                neighbors[j] = dist

            # Finding the k-nearest neighbors
            nearest_neighbors = sorted(neighbors.items(), key=lambda x: x[1])[:self.K]
            nearest_neighbors = [neighbor[0] for neighbor in  nearest_neighbors] # List containing only indexes of nearest neighbors

            # Getting the predicted label for the new point
            pred_label = self.get_weighted_label(nearest_neighbors)

            # Checking to see if predicted label is correct
            if pred_label == input_y[i]:
                correct_predictions += 1

        # Returning the accuracy of the model
        # TODO: Return a list of predictions instead of the accuracy
        return correct_predictions / len(input_x)


    # Predicts the label of a point given a list of IDs of the nearest neighbors to that point
    # NOTE: Assumes labels are the last value in the row
    def get_weighted_label(self, nearest_neighbors: list):
        label_votes = {}

        # Counting the labels for each
        for i in nearest_neighbors:
            label = self.training_data_y[i] # Label value
            
            # Adding label vote to the dictionary
            if label_votes.__contains__(label):
                label_votes[label] = label_votes[label] + 1
            else:
                label_votes[label] = 1

        # Returning the label with the most votes (sorted in ascending order)
        return sorted(label_votes.items(), key=lambda x: x[1])[-1][0]