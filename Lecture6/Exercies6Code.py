import IPython.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

%matplotlib inline

X = np.arange(100)
y = X + np.random.RandomState(0).uniform(0, 20, 100)
# This visualizes the data as a scatter plot where each data point is a single dot
plt.scatter(X, y, label="data")
plt.xlabel("random data 1")
plt.ylabel("random data 2")
# This line enables the little box which shows the labels you set on the shown data
plt.legend()
plt.show()

def convert_percentage(string):
    number_as_string = string[:-1]
    number = float(number_as_string)
    return number / 100

def convert_float(string):
    try:
        return float(string)
    except ValueError:
        return None

column_names = ["beverage-category", "beverage", "beverage-prep", "calories",
    "fat-total", "fat-trans", "fat-saturated", "sodium", "carbohydrates",
    "cholesterol", "fibre", "sugars", "protein", "vitamin-a", "vitamin-c",
    "calcium", "iron", "caffeine"]

df = pd.read_csv("starbucks.csv", skiprows=1, names=column_names)

for col in ["vitamin-a", "vitamin-c", "calcium", "iron"]:
    df[col] = df[col].map(convert_percentage)
for col in ["fat-total", "caffeine"]:
    df[col] = df[col].map(convert_float).fillna(0)
    

x = np.arange(50)
y = x + np.random.RandomState(0).uniform(0, 15, size=50)
tmp_df = pd.DataFrame(zip(x, y), columns=["A", "B"])
tmp_df.plot("A", "B")

# The following three lines just make a dataframe with random data for us to plot. They are just for demonstration.
x = np.arange(50)
y = x + np.random.RandomState(0).uniform(0, 15, size=50)
tmp_df = pd.DataFrame(zip(x, y), columns=["A", "B"])
# Here is the important part:
tmp_df.plot.scatter("A", "B")


# Try plotting carbohydrates on the x-axis and calories on the y-axis as a scatter plot.
test_df = pd.DataFrame(zip(df['carbohydrates'], df['calories']), columns=["car", "cal"])
test_df.plot.scatter("car", "cal")
#df.plot.scatter("carbohydrates", "calories")


x = np.arange(50)
y = x + np.random.RandomState(0).uniform(0, 15, size=50)
tmp_df = pd.DataFrame(zip(x, y), columns=["A", "B"])
sns.regplot(x="A", y="B", data=tmp_df, ci=None)


# Fill in this block to draw a scatter plot of the carbohydrates and calories features with a regression line using seaborn
sns.regplot(x="carbohydrates", y="calories", data=df, ci=None)


# Fill in this block using pandas and sklearn to draw a scatter plot and a line representing a linear regression. Use `carbohydrates` for the x-axis and `calories` for the y-axis.

import sklearn.linear_model
model = sklearn.linear_model.LinearRegression()
model.fit(df[["carbohydrates"]], df["calories"])
#model.intercept_

plot = df.plot.scatter("carbohydrates", "calories")
plot.axline((0, model.intercept_), slope = model.coef_[0])



#### BAR

# Fill in this block to draw a bar plot of the element counts of the of the different groups in the `beverage` feature. You can use `Series.value_counts` to get the counts.
#becount = df['beverage'].value_counts()
#df.plot.bar("beverage", becount)
df["beverage"].value_counts().plot.bar()

# Fill in this block to group the dataset by the beverage-category feature and make a bar plot of the different group means.
#df['beverage-category'].value_count.mean().plot.bar()
df.groupby("beverage-category").mean(numeric_only=True).plot.bar(y="carbohydrates")


# Fill in this block to draw a bar plot of the means of the carbohydrate values for the beverage-category groups sorted by the means, highest first.
means = df.groupby("beverage-category").mean(numeric_only=True)
means.loc[means.sort_values("carbohydrates", ascending=False).index].plot.bar(y="carbohydrates")


##### Histograms
# Try plotting the calories column as a histogram
df["calories"].plot.hist()

# Try drawing a seaborn histplot of the calories feature.
sns.histplot(df["calories"], kde=True)


###### Box plots
# Fill in this block to draw as boxplot of all the dataset features.
# Hint: If you show the labels on the x-axis you can rotate them like this: `plot.set_xticklabels(plot.get_xticklabels(), rotation=90)`
# Another way of drawing the plot which makes the text of the labels readable is to pass `vert=False` to `DataFrame.boxplot` in order to draw the boxes horizontally.
plot = df.boxplot()
labels = plot.get_xticklabels()
_ = plot.set_xticklabels(plot.get_xticklabels(), rotation=90)


##### Faceting
# Fill in this block to draw a figure with three box plots, one showing those features with medians above 30, one for those with medians between 2 and 30, and one for those with medians below 2.
# Show the box plots horizontally. Use `figsize=(width, height)` to set the dimensions of the figure if you need it to be larger than default.
median = df.median(numeric_only=True)
figure, axs = plt.subplots(1, 3, figsize=(25, 10))
large_value_features = (median > 30).map(lambda x: x if x else None).dropna().index
middle_value_features = ((median > 2) & (median <= 30)).map(lambda x: x if x else None).dropna().index
small_value_features = (median <= 2).map(lambda x: x if x else None).dropna().index
df[large_value_features].boxplot(vert=False, ax=axs[0])
df[middle_value_features].boxplot(vert=False, ax=axs[1])
df[small_value_features].boxplot(vert=False, ax=axs[2])

##### Heatmaps
# Fill in this block to draw a heatmap of the correlation matrix of the dataset using matplotlib
correlations = df.corr(numeric_only=True)
plt.imshow(correlations, cmap="Reds_r")
plt.colorbar()
plt.xticks(range(correlations.shape[0]), correlations.columns, rotation=90)
plt.yticks(range(correlations.shape[0]), correlations.columns)
plt.show()