import pandas as pd
import plotnine as p9
import datetime
import numpy as np
import sklearn.preprocessing

df = pd.read_csv("path", encoding="latin-1")
df.columns ## it only showed one columns, so that is just wrong.
# so some preprocessing is needed
df = pd.read_csv("path", encoding="latin-1", skiprows=2) ## to remove the weird title and empty row
df = pd.read_csv("path", encoding="latin-1", skiprows=2, names=["add column names"]) ## this makes it easier to work with

df.index.values[:20] # gives dates in a string, but we want to have them as datetime objects

def convert_date(v):
    try:
        year, quarter = v.split("K") # splits on the char "K", but it does take the "K" with the split
        month = {"1": 1, "2": 4, "3": 7, "4": 10} 
        d = datetime.datetime(year=year, month=month[quarter], day=1)
        return d
    except ValueError as ex:
        print(f"waring :", ex)
        return None

convert_date("1991K1") # this works, but the df still have some problems

df.index = df.index.map(convert_date) # gives some errors because it has some weird date that is not used
df = df.dropna(how="all") # this only removes the rows that has no info.
df.index = df.index.map(convert_date) # this is fine now

df.iloc[0] ## this shows that the people who made this dataset, have choice to show that not having data as "..", so we need to change that

df = df.replace("..", np.nan)

df.info() ## if you look at the dtype and sees object it normal means that the type is of string

df = df.applymap(float) # use df.map more.


# this something you could to do to fill out some missing data, but you have to check if it then fit into the rest of the dataset
df["cage-egg-post"] = df["cage-egg"].fillna(df["cage-eggs"].mean()) # in this case, it does not really fit

# if you can see a logic to the data, then you could use forward and backward filling.
df["cage-eggs"].ffill() # if this finds a missing data it will take the data from the row before it
df["cage-eggs"].bfill() # if this finds a missing data it will take the data from the row after it

# so backward filling is best for this 
df["cage-egg-post"] = df["cage-eggs"].bfill()

df["cage-egg"].interpolate(method="linear", limit_direction="both") # this does the same as the back/forward does, but you make pandas do the whole thing
# this works some of the time, but if the data is more weird, then it may not work the way you would want

