import pandas as pd

#df = pd.read_csv("filepath") #gives a big exception, if the encoder cant read the file
#df = pd.read_csv("filepath", encoding="latin1") #can help fix it

l = [1, 2, 3, 4]
def generator(input_list):
    for row in input_list:
        yield row

g  = generator(l)
next(g) #print


import csv
with open("filepath to csv") as fp:
    reader = csv.DictReader(fp) #reads the csv into a dict
    rows = [l for l in reader] #makes all rows into a list
    # all of the row elements is a dict that has a key and a value

len(rows)


#2017 - int(rows[0]["dog-birth-year"])
dog_ages = [2017 - int(r[0]["dog-birth-year"]) for r in rows]
import statistics as stat
stat.mean(dog_ages)
stat.median(dog_ages)
stat.mode(dog_ages)
min(dog_ages)
max(dog_ages)


import collections as coll
c = coll.Counter()
for r in rows:
    c[r["dog-race"]] += 1

c.most_common()

c_gender = coll.Counter()
for r in rows:
    c_gender[r["owner-gender"]] += 1

