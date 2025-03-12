import pandas as pd
import numpy as np
import re

column_names = ["beverage-category", "beverage", "beverage-prep", "calories",
    "fat-total", "fat-trans", "fat-saturated", "sodium", "carbohydrates",
    "cholesterol", "fibre", "sugars", "protein", "vitamin-a", "vitamin-c",
    "calcium", "iron", "caffeine"]

df = pd.read_csv("./starbucks.csv", skiprows=1, names=column_names)

# Show the first five rows
# print(df.head(5))

# Display a summary of the different features
# print(df.describe())


# Define a function which selects and returns the column containing saturated fat from the starbucks dataset
def get_saturated_fat():
    # fill in the function
    return df["fat-saturated"]

# print(get_saturated_fat())

# Comparisons
def get_values(dataframe, columnName):
    # Fill in the function to select all the rows where `fat-total` is larger than 3
    try:
        # print(dataframe["fat-total"])
        return dataframe[dataframe[columnName] > 3]
    except TypeError:
        # print("\ntest\n")
        dataframe = set_non_float_values(dataframe, columnName) # Makes string into number
        dataframe[columnName] = convert_to_floats(dataframe, columnName) # change the now converted dataframe column to floats
        df = dataframe # save the dataframe
        return get_values(df, columnName)
        
def is_float(v):
    try:
        float(v)
        return True
    except ValueError:
        return False
    
# def change_whitespace(elem):
#     result = re.sub(r'(\d)\s(\d)(?!\d)', r'\1.\2', elem)
#     return elem
    
def set_non_float_values(dataframe, column):
    # fill in the function and return a dataframe with zeroes instead of non-convertable values
    mask = dataframe[column].map(is_float)
    dataframe.loc[~mask, column] = 0
    print(mask)
    return dataframe

def convert_to_floats(dataframe, column):
    # Fill in the function to convert values in the given column to floats
    return dataframe[column].astype(float) 


print(get_values(df, "fat-total"))


# Summary statistics
def get_mean(dataframe, column):
    # Fill in this function to get the mean of the given column
    return dataframe[column].mean()

def get_median(dataframe, column):
    # Fill in this function to get the median of the given column
    return dataframe[column].median()

# Fill in this block to get the correlation matrix for the `calories`, `sugars`, `fibre`, and `sodium` features.
def get_correlation_matrix(dataframe):
    # Fill in the function to get the correlation matrix of the `calories`, `sugars`, `fibre`, and `sodium` features
    return dataframe[["calories", "sugars", "fibre", "sodium"]].corr()

def standard_deviation(dataframe, column):
    # Fill in the function to return the standard deviation of `column` from the `dataframe`
    return dataframe[column].std()