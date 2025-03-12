import pandas as pd
import


X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df[["carbi"]], df) 
# this split the data os that we get a sample while also have the whole dataset, 
# so that we can see if the can say something about the whole dataset

model = sklearn.linear_model.LinearRegression()
model.fit(X_train, y_train)
# now that it is fitted, we can begain to infer something about the data
preds = model.predict(X_test)
# each number is a prediction. (i think)
sklearn.metrics.r2_score(y_test, preds) # gives 0.39


# classifation
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df[["carbi, choles"]], df["bevargage-cat"], test_size=0.1)
model = sklearn.´linear_model.RandomForestClassifier()
