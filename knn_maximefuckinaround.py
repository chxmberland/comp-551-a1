import pandas as pd
import numpy as np

# Finds the euclidean distance between two rows in a Pandas DataFrame.
# It treats the values in each column of the rows as a point in a dimension in space.
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

    # Predicts the labels of samples in the test dataset and returns them as a list
    def predict(self, test_data) -> list:

        predictions = []
        pred_probs = []

        print("Testing data shap:", test_data.shape)

        # Looping through each point in the test dataset
        for i in range(0, len(test_data)):
            test_row = test_data[i]

            # Dictionary to hold the distances to each other point
            neighbors = {}

            # Getting the distance between current test point and all training points
            # j being the index of each training row
            for j in range(0, len(self.x)):
                train_row = self.x[j]

                # Taking the rows to find the distance between the two
                dist = self.dist_fn(test_row, train_row)
                neighbors[j] = dist

            # Finding the k-nearest neighbors
            nearest_neighbors = sorted(neighbors.items(), key=lambda x: x[1])[:self.K] # sorted list of tuples
            nearest_neighbors = [x[0] for x in  nearest_neighbors] # List containing only indexes of nearest neighbors

            print("self.y.shape = ", self.y.shape)

            pred_classes = [self.y[i] for i in nearest_neighbors]

            class_probs = pd.Series(pred_classes).value_counts(normalize = True)

            # Getting the predicted label for the new points
            predictions.append(class_probs.index.to_list()[0])

            pred_probs.append(class_probs.iloc[0])

        return predictions, pred_probs



#accuracy = np.sum(predictions == y_test)/y_test.shape[0]
