import pandas as pd

## ----- DATASET 1 ----- ##

# Loading dataset
nhanes = pd.read_csv("dataset1/NHANES_age_prediction.csv")
nhanes.head()
nhanes.info()

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

# Check for missing values
print(nhanes.isnull().sum())

## Gender
print(nhanes.groupby("age_group")["Gender"].value_counts(normalize = True))
# No apparent impact of gender upon age group

## Fitness
print(nhanes.groupby("age_group")["Fitness"].value_counts(normalize = True))

## ----- DATASET 2 ----- ##

# Loading dataset
bcw = pd.read_csv("dataset2/breast-cancer-wisconsin.csv")

# Cleaning dataset
bcw.dropna()