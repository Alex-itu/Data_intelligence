import pandas as pd

df= pd.read_csv("pathToData")
df["beverage"].unique() # this will give all the unique values in there are in a column
df["new-variable"] = df["beverage-prep"].isin(["short", "Tal"])

def to_float(v):
    try:
        return float(v)
    except ValueError:
        return None
df["fat-total"] = df["fat-total"].map(to_float)
df["fat-total"].isna() # gives to to those that has no data in a column row
df["fat-total"].isna().sum # to see how many true there is

df["fat-total"].dropna() # remove the problemate value or lake of it. not too good to use
df["fat-total"].dropna(how="all") #

df["fat-total"].fillna(110) # find and replace. finds the column with no data and then replace it with 110

df.groupby("beverage")["calories"].mean() # groupby looks at relatetionship between data
df.groupby("beverage").agg({"calories": "mean", "fat-total": "median"})

import statistics
def func(v):
    return statistics.mean(v)

df.groupby(["beverage", "beverage-prep"])["calories"].mean()

df.plot() # this is used for Visualization
df["calories"].plot.hist() # for numberic values
df["beverage"].value_counts().plot.bar() # for catoriegry values


#df_dogs["dog-race"].value_counts().plot.bar() # this gave to many data points in the X axel

df.plot.box() # gives a box plot
#df.plot.box(numeric_only=true)
