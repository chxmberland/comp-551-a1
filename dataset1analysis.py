import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from knn_maximefuckinaround import KNN, euclidean_distance


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
nhanes_target = nhanes["age_group"]

print(nhanes_m1.head(), "\n")

## Step 1: splitting data into train, validation, test, roughly 50%, 25%, 25%

X_train, X_test, y_train, y_test = train_test_split(
    nhanes_m1, nhanes_target, test_size = 0.25, random_state = 21

)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size = 0.33, random_state=22
)

print("Training features array dimensions:", X_train.shape)
print("Training target array dimensions:", y_train.shape, "\n")

print("Validation features array dimensions:", X_valid.shape)
print("Validation target array dimensions:", y_valid.shape, "\n")

print("Test features array dimensions:", X_test.shape)
print("Test target array dimensions:", y_test.shape, "\n")

X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

## Step 2: Tuning hyperparameter for KNN (testing for K)

model_1 = KNN(K = 5)

model_1.fit(X_train, y_train)

print(y_train.shape)

m1_preds, m1_probs = model_1.predict(X_test)

#accuracy = np.sum(m1_preds == y_test)/y_test.shape[0]

#print(accuracy)
    




