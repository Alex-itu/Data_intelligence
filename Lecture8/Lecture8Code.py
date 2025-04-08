import pandas as pd
import sklearn.feature_extraction
import sklearn.metrics
import nltk
import sklearn.naive_bayes
import sklearn
# import
# import
import collections
import plotnine as p9
import numpy as np

nltk.download("punkt")

df = "get csv data"

s = df["reviewText"].iloc[0]

s.split() # split on spaces on default
 # get things like "to." which we dont want

tokenizer = nltk.tokenize.TweetTokenizer()
tokens = tokenizer.tokenize(s)
    # now it splits like we want, "to." becomes "to", "."

Lemmatizer = nltk.wordnet.WordNetLemmatizer()
Lemmatizer.Lemmatize(tokens[0])

vectorizer = sklearn.feature_extraction.text.CountVectorizer()
vectorizer.fit_transform(["this is an example"]).toarray()

df["tokens"] = df["reviewText"].str.lower().fillna("").map(tokenizer.tokenize).map(lambda sentence: [Lemmatizer.lemmatize(token) for token in sentence])
bow = vectorizer.fit_transform(df["tokens"].str.join(" "))
 # 
 
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(bow, df["overall"], test_sizer=0.1, random_state=0)
model = sklearn.naive_bayes.MultinomialNB() # good for text prosseging 

model.fit(X_train, y_train)
preds = model.predict(X_test)

print(sklearn.metrics.classification_report(y_test, preds))

p9.ggplot(df) + p9.aes(x="overall") + p9.geom_bar()

g = df.groupby("overall")
balanced_df = g.apply(lambda x: x.sample(g.size().min())).reset_index(drop=True)
# filter out the data set os that all catagory has the same amount of data, this size here is min size

vectorizer_b = sklearn.feature_extraction.text.CountVectorizer()
bow_b = vectorizer_b.fit_transform(balanced_df["tokens"].str.join(" "))

model_b = sklearn.naive_bayes.MultinomialNB()

X_train_b, X_test_b, y_train_b, y_test_b = sklearn.model_selection.train_test_split(bow, balanced_df["overall"], test_sizer=0.1, random_state=0)
model_b.fit(X_train_b, y_train_b)
preds_b = model_b.predict(X_test_b)

print(sklearn.metrics.classification_report(y_test_b, preds_b))


counter = collection.Counter()
balanced_df["tokens"].map(counter.update)
counter.most_commen(30)
# there is a lot of stopword we could remove

stopwords = set([w for w, _ in counter.most_common(30)])

balanced_df["filtered_tokens"] = balanced_df["tokens"].map(lambda sentence: [token for token in sentance if token not in stopwords])

balanced_df["filtered_tokens"]

_vectorizer = sklearn.feature_extraction.text.CountVectorizer()
_bow = _vectorizer.fit_transform(balanced_df["filtered_tokens"].str.join(" "))

_model = sklearn.naive_bayes.MultinomialNB()

_X_train, _X_test, _y_train, _y_test = sklearn.model_selection.train_test_split(_bow, balanced_df["overall"], test_sizer=0.1, random_state=0)
_model.fit(_X_train, _y_train)
_preds = _model.predict(_X_test)

print(sklearn.metrics.classification_report(_y_test, _preds))


vectorizer_n = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(1, 2))
# and then do all the print shit

features = vectorizer_n.get_feature_names_out()
model_n.feature_log_prob_
features[np.atgsort(-model.feature_log_prob_)[:, :30]]