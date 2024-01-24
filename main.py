import pandas as pd
import numpy as np
from knn import KNN, split_data

## ----- DATASET 1 ----- ##

# Loading dataset
nhanes = pd.read_csv("dataset1/NHANES_age_prediction.csv")

# Comes out to true, so no duplicates
print(nhanes.drop_duplicates().shape == nhanes.shape)
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

### ----- Summary stats ----- ###

print("----- DATASET 1 -----\n")

# Check for missing values
print("\n----- NUMBER OF NULL COLUMNS IN NHANES -----\n")
print(nhanes.isnull().sum())

# Gender
print("\n----- GROUPING DATASET BY GENDER -----\n")
print(nhanes.groupby("age_group")["Gender"].value_counts(normalize = True))
# No apparent impact of gender upon age group

# Fitness
print("\n----- GROUPING DATASET BY AGE GROUP -----\n")
print(nhanes.groupby("age_group")["Fitness"].value_counts(normalize = True))

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

# ----- TRAINING THE MODEL -----#
print("\n----- TRAINING ON DATASET TWO -----\n")

# Splitting into 10% test data,10% validation data and 90% training data
test, validation, train = split_data(bcw, 0.1, 0.1)
print("From a total sample size of " + str(len(bcw)) 
      + ", the dataset was split into training data (" 
      + str(len(train)) + " samples), test data ("
      + str(len(test)) + " samples) and validation data ("
      + str(len(validation)) + ")."
)

# Creating and testing a new model
for k in range(1, 20):
    print("\nTesting model with k value " + str(k))
    model = KNN(k)
    model.fit(train)
    accuracy = model.predict(test)
    print("Got efficacy of " + str(round(accuracy, 3)))