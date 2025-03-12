import pandas as pd
df = pd.read_csv("path to dataset")
pd.Series # like a list
pd.DataFrame # is a collection of a list (good for csv)

df["Row Name"] # gives you the data for that row
# can also do .sum or .std to get that

#this will remove the some of the stupid fails in a dataset, like whitespaces
df = pd.read_csv("path to dataset", skiprows=1, names=column_names)

df.describe()  # this will give you a good metric overview of the dataset

#there is a problem with the dataset where a data is 3 2 instead of 3.2
# so we could write a function that checks if some data can be converted to a float 

def is_float(v):
    try:
        float(v)
        return True
    except ValueError:
        return False
    
mask = df["fat-total"].map(is_float) #check the whole row

df[mask] #prints out all the row that could be converted to float

df[~mask] #prints out all the row that couldnt be converted to float

df.loc[~mask, "fat-total"] = 3.2 
# this will then find the value(s) that couldnt be converted in a row
# the = will the replace the value there was before

df["fat-total"] = df["fat-total"].astype(float) 
#converts the newly change row to a float now