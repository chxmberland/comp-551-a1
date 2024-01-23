import pandas as pd

# DATASET 1
# __________________________________________
nhanes = pd.read_csv("dataset1/NHANES_age_prediction.csv")
nhanes.head()
nhanes.info()

# comes out to true, so no duplicates
nhanes.drop_duplicates().shape == nhanes.shape

nhanes.columns

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


## No missing values!
nhanes.isnull().sum()
nhanes.describe()

# DATASET 2
# __________________________________________

# Subset 1: Breast cancer diagnoses
# Columns
# 1. Sample code number            id number
# 2. Clump Thickness               1 - 10
# 3. Uniformity of Cell Size       1 - 10
# 4. Uniformity of Cell Shape      1 - 10
# 5. Marginal Adhesion             1 - 10
# 6. Single Epithelial Cell Size   1 - 10
# 7. Bare Nuclei                   1 - 10
# 8. Bland Chromatin               1 - 10
# 9. Normal Nucleoli               1 - 10
# 10. Mitoses                      1 - 10
# 11. Class:                       (2 for benign, 4 for malignant)

# Loading dataset
bcw = pd.read_csv("dataset2/breast-cancer-wisconsin.csv")

# Dataset
bcw = bcw.rename(columns = {
    
})