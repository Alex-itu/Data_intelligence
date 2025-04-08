import pandas as pd
df = pd.read_csv("path to data")

df.melt()

df_melted = df.melt(id_vars=["iso2", "year"], value_name="cases")

df_melted["gender"] = df_melted["variable"].map(lambda v: v[7]) # gives the f or m from the values

def convert_age(age_string):
    age_string = age_string[8:]
    l = len(age_string)
    if l == 2:
        return age_string
    if l == 3:
        return "0-14"
    return age_string[:2] + "-" + age_string[2:]    

df_melted["age"] = df_melted["variable"].map(convert_age)

del df_melted["variable"]

