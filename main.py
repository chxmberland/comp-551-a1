import pandas as pd

# DATASET 1
nhanes = pd.read_csv("dataset1/NHANES_age_prediction.csv")

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
breast_cancer_wisconsin = pd.read_csv("dataset2/breast-cancer-wisconsin.csv")