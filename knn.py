import pandas as pd
import numpy as np


# Splits the dataset into training and testing data
# Returns a tuple containing the test data (at index 0) and the training data
def split_data(dataset: pd.DataFrame, test_percentage: float, validation_percentage: float, label_column: int):

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

        # Setting distance function
        self.dist_fn = dist_fn
        self.K = K

    # Memorizes the data
    def fit(self, training_data_x: pd.DataFrame, training_data_y: np.ndarray) -> None:
        self.training_data_x = training_data_x
        self.training_data_y = training_data_y


    # Predicts the labels of samples in the test dataset and returns a percentage accuracy
    def predict(self, input_x: pd.DataFrame) -> int:
        probabilities = []

        # Looping through each point in the test dataset
        for i in range(0, len(input_x)):
            r1 = input_x.iloc[i]

            # Dictionary to hold the distances to each other point
            neighbors = {}

            # Getting the distance between current test point and all training points
            for j in range(0, len(self.training_data_x)):
                r2 = self.training_data_x.iloc[j]
                r2_label = self.training_data_y[j]

                # Getting distance between two points and storing in a dictionary
                dist = self.dist_fn(r1, r2)
                neighbors[j] = (dist, r2_label) # Storing distance and label

            # Sorting the dict to get the K nearest neighbors
            nearest_neighbors = sorted(neighbors.items(), key=lambda x: x[1][0])[:self.K]
            nearest_neighbors = [neighbor[1] for neighbor in  nearest_neighbors] # [(distance, label), ...]

            # Getting the predicted label for the new point
            pos_prediction = self.get_weighted_label(nearest_neighbors)

            # Adding the predicted label to the predictions array
            probabilities.append(pos_prediction)

        # Returning the probabilities that each data point has a positive label
        return probabilities


    # Predicts the label of a point given a list of IDs of the nearest neighbors to that point
    def get_weighted_label(self, nearest_neighbors: list[tuple]):
        total_weight = 0
        pos_weight = 0

        # Counting the labels for each
        for i in nearest_neighbors:

            # Inverse distance, to give each neighbor different voting power
            weight = 1 / (i[0] + 0.0001) # Adding small constant to avoid division by 0
            label = i[1]
            
            # Adding weights to the total class weight
            total_weight += weight
            if int(label) == 1:
                pos_weight += weight

        # Returning the probability the label is positive (1)
        return round(pos_weight / total_weight, 5)
    

    # Gets the prediction accuracy of the KNN model with an integer threshold
    def get_pred_accuracy(self, probabilities: list[int], actual_labels: np.ndarray, pos_threshold: int):
        correct_predictions = 0

        # Looping through each prediction and comparing it to the threshold
        for i in range(0, len(probabilities)):

            # Comparing prediction using the threshold
            probability = probabilities[i]
            actual = actual_labels[i]
            label_prediction = 0 if probability < pos_threshold else 1

            # Checking to see if the prediction was correct
            if actual == label_prediction:
                correct_predictions += 1

        # Returning the proportion of correct predictions
        return correct_predictions / actual_labels.size

            