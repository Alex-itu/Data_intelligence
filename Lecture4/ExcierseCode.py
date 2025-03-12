#pip install --upgrade pip wheel
#pip install pandas scikit-learn matplotlib scipy

import pandas as pd
import numpy as np

def convert_percentage(string):
    number_as_string = string[:-1]
    number = float(number_as_string)
    return number / 100

column_names = ["beverage-category", "beverage", "beverage-prep", "calories",
    "fat-total", "fat-trans", "fat-saturated", "sodium", "carbohydrates",
    "cholesterol", "fibre", "sugars", "protein", "vitamin-a", "vitamin-c",
    "calcium", "iron", "caffeine"]
df = pd.read_csv("starbucks.csv", skiprows=1, names=column_names)
for col in ["vitamin-a", "vitamin-c", "calcium", "iron"]:
    df[col] = df[col].map(convert_percentage)
    
df['fat-total']

from os import error
def convert_float(v):
  if v is None:
    return None
  try:
    return float(v)
  except ValueError:
    return None

def convert_floats_and_fill_missing_values(dataframe, column_name):
  dataframe[column_name] = dataframe[column_name].map(convert_float).fillna(0)
  return dataframe

df.describe()

def get_unique_values(dataframe, column_name):
    return dataframe[column_name].unique()
  
def group_and_get_mean(dataframe):  
  return dataframe.groupby(["beverage-prep", "beverage-category"]).mean(numeric_only=True)

def get_max_calories(dataframe):
    # Fill in the function here
    return dataframe.loc[dataframe.groupby("beverage-prep")["calories"].idxmax()]
  
import scipy.stats
def mean_median_and_trimmed_mean(dataframe):
  # Fill in the function here
  trim_mean = lambda x, *args: scipy.stats.trim_mean(x, 0.1)

  return dataframe.groupby("beverage-category").agg({"calories": ["mean", "median", trim_mean]})

df["calories"].plot.hist()

df["beverage-category"].value_counts().plot.bar()

df.plot.scatter(x="sugars", y="calories")

df_paper.plot(x="year", y="Cielab b*")