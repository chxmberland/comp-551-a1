import pandas as pd
import numpy as np
from knn import KNN, split_data
from dt import DecisionTree
from sklearn.model_selection import train_test_split


# DATASET 1
# __________________________________________

### ----- Load in and clean up ----- ###

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


### ----- Summary stats ----- ###

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


## Variables to consider in KNN:
#
# - Fitness (categorical -- 2 levels)
# - BMI (cont.)
# - Blood glucose (cont.)
# - Diabetic (categorical -- 3 levels)
# - Oral (cont.)
# - Insulin (cont.)


### ----- Apply KNN ----- ###

# Model 1: continuous variables only for simplicity

nhanes_m1 = nhanes[["BMI", "Blood_glucose", "Oral", "Insulin"]]



## ----- DATASET 2 ----- ##

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

# TODO: Remove noise features


# ----- TRAINING THE MODEL -----#
#print("\n----- TRAINING ON DATASET TWO -----\n")

# Splitting into 10% test data,10% validation data and 90% training data
test, validation, train = split_data(bcw, 0.1, 0.1)
print("From a total sample size of " + str(len(bcw)) 
      + ", the dataset was split into training data (" 
      + str(len(train)) + " samples), test data ("
      + str(len(test)) + " samples) and validation data ("
      + str(len(validation)) + ")."
)

# Tuning the hyperparameter
best_k = 0
best_accuracy = 0
for k in range(1, 11):
    print("\nTesting model with k value " + str(k))

    # Creating a new KNN model
    model = KNN(k)
    model.fit(train)
    new_accuracy = model.predict(validation) # Predict using validation data
    print("Got efficacy of " + str(round(new_accuracy, 3)))
    
    # Setting k and accuracy variables
    if new_accuracy > best_accuracy:
        best_accuracy = new_accuracy
        best_k = k

print("Using k-value " + str(k))

# Testing with test data
model = KNN(best_k)
accuracy = model.predict(test) # Predicting with test data
print(accuracy)

#DECISION TREE
#
#
#

print("\n----- TRAINING ON DATASET ONE -----\n")

dataset_size = nhanes.shape[0]
num_cols = nhanes.shape[1]
#NEED TO FIGURE OUT HOW 
nhanes = nhanes.to_numpy()

#replace 'male' and 'female' with 0 and 1 respectively
nhanes[:, 1] = np.where(nhanes[:, 1].astype(str) == 'Adult', 0, 1).astype(int)
#change float col to int
nhanes[:, -1] = (nhanes[:, -1] * 100).astype(int)


#change all cols to int
for col in range(num_cols):
    if col == 1 or col == num_cols - 1:
        continue
    else:
        nhanes[:, col] = (nhanes[:, col]).astype(int)

print(nhanes)
c = 0
#for i in range(num_cols):
#    for j in range(dataset_size):
#        if type(nhanes[j,i]) != int:
#            print(type(nhanes[j,i]))
#            c += 1

    #print(type(nhanes[0,i]))
#bcw.set_index(pd.Index([i for i in range(bcw.shape[0])]))
inds = np.random.permutation(dataset_size)
test_proportion = 0.25
test_size = int(test_proportion*dataset_size)
train_size = dataset_size-test_size
#print(train_size, test_size)
'''x, y = bcw.iloc[:,:-1], bcw.iloc[:,-1]

x_train, y_train = x.iloc[inds[:train_size],:], y.iloc[inds[:train_size]]
x_test, y_test = x.iloc[inds[train_size:],:], y.iloc[inds[train_size:]]'''

want_to_select = [True for _ in range(num_cols)]
#remove ID and age label from X features
want_to_select[0] = False
want_to_select[1] = False
x, y = nhanes[:,np.array(want_to_select)], nhanes[:,1]

x_train, y_train = x[inds[:train_size]], y[inds[:train_size]]
x_test, y_test = x[inds[train_size:]], y[inds[train_size:]]

print(x_train)
print(y_train)
print(x_test)
print(y_test)

#print(x_train, y_train)
#print(x_test, y_test)

DTmodel = DecisionTree()
DTmodel.fit(x_train, y_train)
predictedClassProbs = DTmodel.predict(x_test)
#print(predictedClassProbs)
predictedClasses = []
for v in predictedClassProbs:
    maxp = -1
    maxIndex = -1
    for i in range(len(v)):
        if v[i] > maxp:
            maxp = v[i]
            maxIndex = i
    predictedClasses.append(maxIndex)

#print(predictedClasses)

accurate_preds = 0
for i in range(len(predictedClasses)):
    if predictedClasses[i] == y_test[i]:
        accurate_preds += 1

accuracy = accurate_preds / len(predictedClasses)
print(accuracy)

print("\n----- TRAINING ON DATASET TWO -----\n")
#print(bcw)
dataset_size = bcw.shape[0]
bcw = bcw.to_numpy().astype(int)
#bcw.set_index(pd.Index([i for i in range(bcw.shape[0])]))
inds = np.random.permutation(dataset_size)
test_proportion = 0.25
test_size = int(test_proportion*dataset_size)
train_size = dataset_size-test_size
#print(train_size, test_size)
'''x, y = bcw.iloc[:,:-1], bcw.iloc[:,-1]

x_train, y_train = x.iloc[inds[:train_size],:], y.iloc[inds[:train_size]]
x_test, y_test = x.iloc[inds[train_size:],:], y.iloc[inds[train_size:]]'''

x, y = bcw[:,:-1], bcw[:,-1]

x_train, y_train = x[inds[:train_size]], y[inds[:train_size]]
x_test, y_test = x[inds[train_size:]], y[inds[train_size:]]

print(x_train)
print(y_train)
print(x_test)
print(y_test)

#print(x_train, y_train)
#print(x_test, y_test)

DTmodel = DecisionTree()
DTmodel.fit(x_train, y_train)
predictedClassProbs = DTmodel.predict(x_test)
#print(predictedClassProbs)
predictedClasses = []
for v in predictedClassProbs:
    maxp = -1
    maxIndex = -1
    for i in range(len(v)):
        if v[i] > maxp:
            maxp = v[i]
            maxIndex = i
    predictedClasses.append(maxIndex)

#print(predictedClasses)

accurate_preds = 0
for i in range(len(predictedClasses)):
    if predictedClasses[i] == y_test[i]:
        accurate_preds += 1

accuracy = accurate_preds / len(predictedClasses)
print(accuracy)


#TODO: CREATE VALIDATION SET AND FUCK AROUND WITH MAX DEPTH AND DIFF COST FUNCTIONS FOR REPORT
