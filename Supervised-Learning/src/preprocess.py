"""Preprocess.ipynb

Original file is located at
    https://colab.research.google.com/drive/1ppEI3MSPQ0poQ8uaddjNL6jG_sruPjM1

**Data Preprocessing for GaIB's Supervised Learning Task**

*Preprocess the `churn modelling` dataset to separate categorical and numerical feature. In the dataset, the numerical features are:*
1.   Credit Score
2.   Age
3. Tenure
4. Balance
5. Number of Products
6. Estimated Salary
*The rest are categorical features, and the last one is the label of the dataset.*
"""

import pandas as pd

# Basic
def read_dataset(csv):
    dataset = pd.read_csv(csv)
    return dataset


def clean_dataset(dataset):
    # Drop first three unused columns
    dataset.drop(dataset.columns[0:3], axis=1, inplace=True)
    return dataset


# K-Nearest Neighbor and Logistic Regression
def preprocess_num(csv):
    dataset = read_dataset(csv)
    dataset = clean_dataset(dataset)
    num_col_idx = [1, 2, 7, 8]
    dataset.drop(dataset.columns[num_col_idx], axis=1, inplace=True)

    return dataset


# dataset = preprocess_num("/content/dataset.csv")


# Iterative Dichotomiser 3
def preprocess_cat(csv):
    dataset = read_dataset(csv)
    dataset = clean_dataset(dataset)
    cat_col_idx = [idx for idx in range(11) if idx not in [1, 2, 7, 8, 10]]
    dataset.drop(dataset.columns[cat_col_idx], axis=1, inplace=True)

    # Convert string categories to integer values
    # Country Mappings
    countries_map = {"France": 1, "Spain": 2, "Germany": 3}
    # Gender Mappings
    gender_map = {"Male": 0, "Female": 1}

    # Replace the value in the dataset with the maps
    dataset.replace({"Geography": countries_map, "Gender": gender_map}, inplace=True)

    return dataset


# dataset = preprocess_cat("/content/dataset.csv")

# print(dataset)
