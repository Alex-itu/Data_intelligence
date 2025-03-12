import pandas as pd
import plotnine as p9
import seaborn as sna

%matlab inline

df = pd.read_csv()

print(df)

plot = df["culmen_length_mm"].plot.hist()

sns.histplot(data=df, x="culmen_length_mm")

sns.countplot(data=df, x="species")

sns.boxplot(data=df, y="culmen_length_mm")

sns.scatterplot(data=df, x="flipper_lengrh_mm", y="body_mass_g")

sns.scatterplot(data=df, x="flipper_lengrh_mm", y="body_mass_g", hue="species")


import sklearn.linear_model
model=sklearn.linear_model.LinearRegression()
model.fit(df[["flipper"]].fillna(0), df["body"].fillna(0))
model.intercept_
model.coef_

plotsca = sns.scatterplot(data=df, x="flipper_lengrh_mm", y="body_mass_g", hue="species")
plotsca.axline((0, model.intercept_), slope=model.coef_[])

p9.ggplot(df) + p9.aes(x="flipper", y="body") + p9.geom_point()

p9.ggplot(df) + p9.aes(x="flipper", y="body") + p9.geom_point() + p9.facet_wrap("species")

p9.ggplot(df) + p9.aes(x="flipper", y="body", color="species") + p9.geom_point()

p9.ggplot(df) + p9.aes(x="flipper", y="body") + p9.geom_point() + p9.facet_wrap("sex")