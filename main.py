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

# Changing classes column to integers
nhanes['age_group'] = nhanes['age_group'].replace({"Adult": 0, "Senior": 1})
print(nhanes)
og_nhanes = nhanes

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


#util functions
def evaluate_acc(pred_y, real_y):
    accurate_preds = 0
    for i in range(len(pred_y)):
        if pred_y[i] == real_y[i]:
            accurate_preds += 1

    train_accuracy = accurate_preds / len(pred_y)
    return train_accuracy

# TODO: Remove noise features

# ----- TRAINING THE MODEL -----#

# ----- TRAINING ON DATASET ONE ----- #

print("\n----- TRAINING ON DATASET ONE -----\n")

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

    # Creating a new KNN model
    model = KNN(k)
    model.fit(train_x, train_y)
    new_accuracy = model.predict(validation_x, validation_y) # Predict using validation data
    print("Got efficacy of " + str(round(new_accuracy, 3)))
    
    # Setting k and accuracy variables
    if new_accuracy > best_accuracy:
        best_accuracy = new_accuracy
        best_k = k

print("\nUsing k-value on test data " + str(best_k))

# Testing with test data
model = KNN(best_k)
model.fit(train_x, train_y)
accuracy = model.predict(test_x, test_y) # Predicting with test data
print(str(round(accuracy, 2)))

# ----- TRAINING DATASET TWO ----- #

print("\n----- TRAINING ON DATASET TWO -----\n")

# Splitting into 10% test data,10% validation data and 90% training data
test_x, test_y, validation_x, validation_y, train_x, train_y = split_data(bcw, 0.1, 0.1, bcw.shape[1] - 1)
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

    # Creating a new KNN model
    model = KNN(k)
    model.fit(train_x, train_y)
    new_accuracy = model.predict(validation_x, validation_y) # Predict using validation data
    print("Got efficacy of " + str(round(new_accuracy, 3)))
    
    # Setting k and accuracy variables
    if new_accuracy > best_accuracy:
        best_accuracy = new_accuracy
        best_k = k

print("\nUsing k-value on test data " + str(best_k))

# Testing with test data
model = KNN(best_k)
model.fit(train_x, train_y)
accuracy = model.predict(test_x, test_y) # Predicting with test data
print(str(round(accuracy, 2)))

#DECISION TREE
#
#
#

print("\n----- DECISION TREE SECTION -----\n")

print("\n----- TRAINING ON DATASET ONE -----\n")

dataset_size = nhanes.shape[0]
num_cols = nhanes.shape[1]
nhanes = og_nhanes.to_numpy()

#change float col to int
nhanes[:, 5] *= 10#.astype(int)
nhanes[:, -1] *= 100#.astype(int)

for col in range(num_cols):
    nhanes[:, col] = (nhanes[:, col]).astype(int)

inds = np.random.permutation(dataset_size)
test_proportion = 0.25
validate_proportion = 0.25
test_size = int(test_proportion*dataset_size)
validate_size = int(validate_proportion*dataset_size)
train_size = dataset_size-test_size-validate_size


want_to_select = [True for _ in range(num_cols)]
#remove ID and age label from X features
want_to_select[0] = False
want_to_select[1] = False
want_to_select[2] = False
x, y = nhanes[:,np.array(want_to_select)], nhanes[:,1]

x_train, y_train = x[inds[:train_size]], y[inds[:train_size]]
x_validate, y_validate = x[inds[train_size:train_size+validate_size]], y[inds[train_size:train_size+validate_size]]
x_test, y_test = x[inds[train_size+validate_size:]], y[inds[train_size+validate_size:]]

cost_functions = ["cost_misclassification", "cost_gini_index", "cost_entropy"]
max_max_depth = 10

max_accuracy_function = None
max_accuracy_max_depth = None
max_accuracy = None

for fn in cost_functions:
    for max_depth in range(1,max_max_depth+1):
        DTmodel = DecisionTree(max_depth=max_depth, cost_fn=fn)
        DTmodel.fit(x_train, y_train)
        train_predictedClassProbs = DTmodel.predict(x_test)
        train_predictedClasses = []
        for v in train_predictedClassProbs:
            maxp = -1
            maxIndex = -1
            for i in range(len(v)):
                if v[i] > maxp:
                    maxp = v[i]
                    maxIndex = i
            train_predictedClasses.append(maxIndex)
        
        train_accurate_preds = 0
        for i in range(len(train_predictedClasses)):
            if train_predictedClasses[i] == y_test[i]:
                train_accurate_preds += 1

        train_accuracy = train_accurate_preds / len(train_predictedClasses)
        #print(f'TRAIN ACCURACY ON DATASET ONE OF DECISION TREE WITH COST FUNCTION {fn} AND MAX DEPTH {max_depth} IS {train_accuracy}')

        val_predictedClassProbs = DTmodel.predict(x_validate)
        val_predictedClasses = []
        for v in val_predictedClassProbs:
            maxp = -1
            maxIndex = -1
            for i in range(len(v)):
                if v[i] > maxp:
                    maxp = v[i]
                    maxIndex = i
            val_predictedClasses.append(maxIndex)

        #print(val_predictedClassProbs)
        #print(y_validate)
        val_accuracy = evaluate_acc(val_predictedClasses, y_validate)
        #print(f'VALIDATION ACCURACY ON DATASET ONE OF DECISION TREE WITH COST FUNCTION {fn} AND MAX DEPTH {max_depth} IS {val_accuracy}')

        if max_accuracy is None or val_accuracy > max_accuracy:
            max_accuracy_function = fn
            max_accuracy_max_depth = max_depth
            max_accuracy = val_accuracy

print(f'BEST DECISION TREE MODEL FOR DATASET ONE HAS COST FUNCTION {max_accuracy_function} AND MAX DEPTH {max_accuracy_max_depth} WITH ACCURACY {max_accuracy}')


print("\n----- TRAINING ON DATASET TWO -----\n")

dataset_size = bcw.shape[0]
bcw = bcw.to_numpy().astype(int)

inds = np.random.permutation(dataset_size)
'''test_proportion = 0.25
test_size = int(test_proportion*dataset_size)
train_size = dataset_size-test_size

x, y = bcw[:,:-1], bcw[:,-1]

x_train, y_train = x[inds[:train_size]], y[inds[:train_size]]
x_test, y_test = x[inds[train_size:]], y[inds[train_size:]]'''

test_proportion = 0.25
validate_proportion = 0.25
test_size = int(test_proportion*dataset_size)
validate_size = int(validate_proportion*dataset_size)
train_size = dataset_size-test_size-validate_size

x, y = bcw[:,:-1], bcw[:,-1]

x_train, y_train = x[inds[:train_size]], y[inds[:train_size]]
x_validate, y_validate = x[inds[train_size:train_size+validate_size]], y[inds[train_size:train_size+validate_size]]
x_test, y_test = x[inds[train_size+validate_size:]], y[inds[train_size+validate_size:]]

max_accuracy_function = None
max_accuracy_max_depth = None
max_accuracy = None
for fn in cost_functions:
    for max_depth in range(1,max_depth+1):
        DTmodel = DecisionTree(max_depth=max_depth, cost_fn=fn)
        DTmodel.fit(x_train, y_train)
        train_predictedClassProbs = DTmodel.predict(x_test)
        train_predictedClasses = []
        for v in train_predictedClassProbs:
            maxp = -1
            maxIndex = -1
            for i in range(len(v)):
                if v[i] > maxp:
                    maxp = v[i]
                    maxIndex = i
            train_predictedClasses.append(maxIndex)

        train_accurate_preds = 0
        for i in range(len(train_predictedClasses)):
            if train_predictedClasses[i] == y_test[i]:
                train_accurate_preds += 1

        train_accuracy = train_accurate_preds / len(train_predictedClasses)
        #print(f'TRAIN ACCURACY ON DATASET TWO OF DECISION TREE WITH COST FUNCTION {fn} AND MAX DEPTH {max_depth} IS {train_accuracy}')

        val_predictedClassProbs = DTmodel.predict(x_validate)
        val_predictedClasses = []
        for v in val_predictedClassProbs:
            maxp = -1
            maxIndex = -1
            for i in range(len(v)):
                if v[i] > maxp:
                    maxp = v[i]
                    maxIndex = i
            val_predictedClasses.append(maxIndex)

        val_accuracy = evaluate_acc(val_predictedClasses, y_validate)
        #print(f'VALIDATION ACCURACY ON DATASET TWO OF DECISION TREE WITH COST FUNCTION {fn} AND MAX DEPTH {max_depth} IS {val_accuracy}')

        if max_accuracy is None or val_accuracy > max_accuracy:
            max_accuracy_function = fn
            max_accuracy_max_depth = max_depth
            max_accuracy = val_accuracy

print(f'BEST DECISION TREE MODEL FOR DATASET TWO HAS COST FUNCTION {max_accuracy_function} AND MAX DEPTH {max_accuracy_max_depth} WITH ACCURACY {max_accuracy}')


#TODO: USE VALIDATION SET LIKE THEY DO IN EXAMPLE DT CODE TO PARAMETRIZE MODEL
