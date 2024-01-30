import pandas as pd
import numpy as np
from sklearn import metrics as skm
import matplotlib.pyplot as plt

from knn import KNN, split_data



## ------------ CLEANING: DATASET 1 ------------ ##




nhanes = pd.read_csv("dataset1/NHANES_age_prediction.csv")
nhanes.head()
nhanes.info()

## Checking for duplicates
# comes out to true, so no duplicates
nhanes.drop_duplicates().shape == nhanes.shape

## Rename columns
nhanes = nhanes.rename(columns = {
    "SEQN":"Index",
    "RIDAGEYR":"Age",
    "RIAGENDR":"Gender",
    "PAQ605":"Fitness",
    "BMXBMI":"BMI",
    "LBXGLU":"Blood_glucose",
    "DIQ010":"Diabetic",
    "LBXGLT": "Oral",
    "LBXIN": "Insulin"
})

## Checking for missing values
# no missing values!
nhanes.isnull().sum()
nhanes.describe()

# Wonky value of 7 for 1 row in "Fitness"
nhanes["Fitness"].value_counts()
# Dropping the row:
nhanes = nhanes.drop(nhanes[nhanes["Fitness"] == 7].index)
# Verify:
nhanes["Fitness"].value_counts()

# Changing classes column to integers
nhanes['age_group'] = nhanes['age_group'].replace({"Adult": 0, "Senior": 1})
print(nhanes)

# Summary stats

## Gender
print("Gender proportions by age group:")
print(nhanes.groupby("age_group")["Gender"].value_counts(normalize = True))
# No apparent impact of gender upon age group

## Fitness
print("Fitness levels by age group:")
print(nhanes.groupby("age_group")["Fitness"].value_counts(normalize = True))
# Fitness does appear to predict/depend upon age group

## BMI
nhanes.groupby("age_group")["BMI"].describe()
# not crazy different, but may be signifcant

## Blood Glucose
nhanes.groupby("age_group")["Blood_glucose"].describe()
# the seniors have noticeably higher blood glucose levels

## Diabetic
nhanes["Diabetic"].value_counts()
# Values are 2, 3, 1. 2 means not-diabetic, don't know what 1 and 3 mean
print(nhanes.groupby("age_group")["Diabetic"].value_counts(normalize = True))
# Higher proportion of 1s and 3s among seniors -- prolly big indicator

## Oral
nhanes.groupby("age_group")["Oral"].describe()
# Much higher among seniors rather than adults

## Insulin
nhanes.groupby("age_group")["Insulin"].describe()
# Lower among seniors vs adults

# Model 1: continuous variables only for simplicity

nhanes_m1 = nhanes[["BMI", "Blood_glucose", "Oral", "Insulin"]]



## ------------ CLEANING DATASET 2 ------------ ##



# Loading dataset
bcw = pd.read_csv("dataset2/breast-cancer-wisconsin.csv")

# Removing the first column as it contains ids that we don't need
bcw = bcw.drop(bcw.columns[0], axis=1)

# Creating column names
column_names = ["clump_thickness","cell_uniformity","cell_shape","marginal_adhesion","epithereal_cell_size","bare_nuclei","bland_chromatin","normal_nucleoli","mitoses","class"]
bcw.columns = column_names

# Replacing all '?' characters with NaN
bcw.replace('?', np.nan, inplace=True)

# Converting all rows to numeric values, setting any rows that can't be converted to NaN
bcw = bcw.apply(pd.to_numeric, errors='coerce')

# Dropping all rows with NaN
bcw = bcw.dropna()

# Changing all labels of 2 (benign) to 0 and 4 (malignant) to 1
bcw[bcw.columns[len(bcw.columns) - 1]] = bcw[bcw.columns[len(bcw.columns) - 1]].apply(lambda n: 0 if n == 2 else 1 if n == 4 else n)



# ------------ TRAINING THE MODEL ------------#



# ------------ TRAINING ON DATASET ONE ------------ #



print("\n------------ TRAINING ON DATASET ONE ------------\n")

# Splitting into 10% test data,10% validation data and 90% training data
test_x, test_y, validation_x, validation_y, train_x, train_y = split_data(nhanes, 0.1, 0.1, 1) # Labels are in column 1
print("From a total sample size of " + str(len(bcw)) 
      + ", the dataset was split into training data (" 
      + str(len(train_x)) + " samples), test data ("
      + str(len(test_x)) + " samples) and validation data ("
      + str(len(validation_x)) + ")."
)

best_k = 0
best_accuracy = 0
for k in range(1, 11):
    print("\nTesting model with k value " + str(k))

    # Creating a new KNN model with a new K
    model = KNN(k)

    # Training the model on the training data (memorizing)
    model.fit(train_x, train_y)

    # Deriving the probabilities each data point has a positive label (validation stage)
    probabilities = model.predict(validation_x)

    # Checking the accuracy of the predictions
    new_accuracy, _ = model.evaluate_threshold_acc(probabilities, validation_y, 0.5)

    print("Got accuracy of " + str(round(new_accuracy, 3)))
    
    # Setting k and accuracy variables
    if new_accuracy > best_accuracy:
        best_accuracy = new_accuracy
        best_k = k

print("\nUsing k-value of " + str(k) + " on test data " + str(best_k))

# Testing with test data
model = KNN(best_k)

# Training the model
model.fit(train_x, train_y)

# Predicting labels
probabilities = model.predict(test_x)

# Checking the prediction accuracy with a threshold of 0.5
accuracy, _ = model.evaluate_threshold_acc(probabilities, test_y, 0.5)
print("Got accuracy on test data of " + str(round(accuracy, 2)))

# Computing AUROC
print("\nGetting the AUROC score.")

# Compute ROC curve
fpr, tpr, thresholds = skm.roc_curve(test_y, probabilities)

# Compute AUC
auc = skm.roc_auc_score(test_y, probabilities)

# Plotting
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='darkgrey', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Example')
plt.legend(loc="lower right")
plt.show()



# ------------ TRAINING DATASET TWO ------------ #



print("\n------------ TRAINING ON DATASET TWO ------------\n")

# Splitting into 10% test data,10% validation data and 90% training data
print(bcw)
test_x, test_y, validation_x, validation_y, train_x, train_y = split_data(bcw, 0.1, 0.1, bcw.shape[1] - 1) # Labels are in column 1
print("From a total sample size of " + str(len(bcw)) 
      + ", the dataset was split into training data (" 
      + str(len(train_x)) + " samples), test data ("
      + str(len(test_x)) + " samples) and validation data ("
      + str(len(validation_x)) + ")."
)

best_k = 0
best_accuracy = 0
for k in range(1, 10):
    print("\nTesting model with k value " + str(k))

    # Creating a new KNN model with a new K
    model = KNN(k)

    # Training the model on the training data (memorizing)
    model.fit(train_x, train_y)

    # Deriving the probabilities each data point has a positive label (validation stage)
    probabilities = model.predict(validation_x)

    # Checking the accuracy of the predictions
    new_accuracy, _ = model.evaluate_threshold_acc(probabilities, validation_y, 0.5)

    print("Got accuracy of " + str(round(new_accuracy, 3)))
    
    # Setting k and accuracy variables
    if new_accuracy > best_accuracy:
        best_accuracy = new_accuracy
        best_k = k

print("\nUsing k-value of " + str(k) + " on test data " + str(best_k))

# Testing with test data
model = KNN(best_k)

# Training the model
model.fit(train_x, train_y)

# Predicting labels
probabilities = model.predict(test_x)

# Checking the prediction accuracy with a threshold of 0.5
accuracy, _ = model.evaluate_threshold_acc(probabilities, test_y, 0.5)
print("Got accuracy on test data of " + str(round(accuracy, 2)))

# Computing AUROC
print("\nGetting the AUROC score.")
fpr, tpr, thresholds = skm.roc_curve(test_y, probabilities)

# Compute AUC
auc = skm.roc_auc_score(test_y, probabilities)

# Plotting
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='darkgrey', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Example')
plt.legend(loc="lower right")
plt.show()