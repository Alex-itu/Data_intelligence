import collections

import nltk
# Download a dataset used for tokenization, i.e. splitting our sentences into parts.
nltk.download("punkt")
import numpy as np
import pandas as pd
# This module is only used for loading the data
import datasets

d = datasets.load_dataset("sentiment140", split="train", streaming=True, trust_remote_code=True)

df = pd.DataFrame(((r["text"], r["sentiment"]) for r in d.shuffle(buffer_size=5_000_000, seed=0).take(50000)), columns=["tweet", "sentiment"])
df["sentiment"] = df["sentiment"].map({0: "negative", 4: "positive"})

df.head(20)

plot = df["sentiment"].value_counts().plot.bar()
plot.set_ylabel("count")
plot.set_title("Tweet sentiment counts")


def get_words(string):
    # Fill in the function to split the string into individual tokens. Return a list of tokens.
    tokenizer = nltk.tokenize.TweetTokenizer()
    tokens = tokenizer.tokenize(string)
    return tokens

_words = get_words("It seems we are stuck on the ground in Amarillo. They have put a ground stop for all flights leaving for Denver. Said updates in an hour")
assert isinstance(_words, list), "The returned value is not a list. Check your return statement."
assert len(_words) == 29, "The returned list does not have the expected number of tokens."
assert all(isinstance(w, str) for w in _words), "The elements in the list are not all strings."
assert _words == ['It', 'seems', 'we', 'are', 'stuck', 'on', 'the', 'ground', 'in', 'Amarillo', '.', 'They', 'have', 'put', 'a', 'ground', 'stop', 'for', 'all', 'flights', 'leaving', 'for', 'Denver', '.', 'Said', 'updates', 'in', 'an', 'hour'], "The tokenized list did not have the expected tokens."
assert get_words("This is a url: http://somepage.com - it should be tokenized as a single token") == ['This',  'is', 'a', 'url', ':', 'http://somepage.com', '-', 'it', 'should', 'be', 'tokenized', 'as', 'a', 'single', 'token'], "Tokenizing a string with a url did not give the expected result."
print("Tokenization tests passed!")


def remove_ats(words):
    # Fill in the function to remove adressee indicators from the given list of tokens. Return the filtered list of tokens.
    result = []
    for s in words:
      if "@" not in s:
        result.append(s)   
    return result

cleaned_string = remove_ats(["@someone", "Hello", "there!"])
assert cleaned_string is not None, "The cleaned string was None. Did you forget to return a value from your function?"
assert cleaned_string == ["Hello", "there!"], "The cleaned string didn't match the expected string. Check that you removed only the @word."
cleaned_string = remove_ats(["@someone", "Hello", "there", "@someotherone", "!!!"])
assert cleaned_string == ["Hello", "there", "!!!"], "The cleaned string didn't match the expected string. Check that you removed only the @words."
assert remove_ats(["@someone"]) == [], "The cleaned string didn't match the expected string. Check your return value."
print("@-cleaning tests succeeded!")

def remove_links(words):
    # Fill in the function to remove links from the token list. Return the filtered list of tokens.
    result = []
    for s in words:
      if "http" not in s:
        result.append(s)   
    return result

cleaned_string = remove_links(["This", "is", "a", "link:", "http://somepage.org"])
assert cleaned_string is not None, "The cleaned string was None. Did you forget to return a value from your function?"
assert cleaned_string == ["This", "is", "a", "link:"], "The cleaned string didn't match the expected string. Check that you removed only the link."
cleaned_string = remove_links(["This", "is", "a", "link:", "http://somepage.org", "and", "https://someotherpage.com", "as", "well."])
assert cleaned_string == ["This", "is", "a", "link:", "and", "as", "well."], "The cleaned string didn't match the expected string. Check that you removed only the links."
assert remove_links(["http:link.edu"]) == [], "The cleaned string didn't match the expected string. Check your return value."
print("Link cleaning tests succeeded!")

def lowercase_tokens(words):
    # Fill in this function to lowercase the input series
    return words.str.lower()

lowercased_words = lowercase_tokens(df["tweet"])
assert isinstance(lowercased_words, pd.Series), "The returned value is not a Series. Check your return statement."
assert lowercased_words.shape[0] == 50000, f"The returned Series does not have the expected length: {lowercased_words.shape[0]} != 50000"
assert lowercased_words.iloc[1] == "just got home from church. now time to do laundry and later a nap! ", "The returned Series does not have the expected content."
print("Lowercasing tests passed!")

def get_stopwords(dataframe, column, n_most_common):
    import collections 
    # Fill in the function to return a list of the words most often occurring. Include only the words, not the counts.
    counter = collections.Counter()
    dataframe[column].str.lower().map(get_words).apply(counter.update)
    #dataframe[column].apply(counter.update)
    listOfWordsandCounts = counter.most_common(n_most_common)
    
    result = []
    for word, count in listOfWordsandCounts:
      result.append(word)      

    return result

tmp_stopwords = get_stopwords(df.copy(), "tweet", 50)
assert tmp_stopwords is not None, "The stopwords list is None. Did you forget to return a value from your function?"
assert len(tmp_stopwords) == 50, "Stopwords list doesn't have the expected length (50). Check that you used the `n_most_common` argument."
assert tmp_stopwords[:8] == ["!", ".", "i", "to", "the", ",", "a", "my"], "Stopword list doesn't have the expected content. Check that you lowercased the words before counting them."
print("Stopwords tests succeeded!")
stopwords = set(tmp_stopwords)

def remove_stopwords(words, stopwords):
    # Fill in the function to remove stopwords from the list of tokens.
    result = []

    for word in words:
      if word not in stopwords:
        result.append(word)
    return result
#remove_stopwords(["i'm", "working", "in", "the", "garden"], stopwords)

filtered_string = remove_stopwords(["i'm", "working", "in", "the", "garden"], stopwords)
assert filtered_string is not None, "The filtered string was None. Did you forget to return a value from your function?"
assert filtered_string == ["working", "garden"], "The filtered string didn't match the expected string. Check your filtering."
print("Filtering tests succeeded!")

def clean_strings(dataframe, column_to_clean, cleaned_column):
    # Fill in the function to remove both @words and links from the given dataframe
    dataframe[cleaned_column] = dataframe[column_to_clean]\
        .str.lower()\
        .map(get_words)\
        .map(lambda words: remove_stopwords(words, stopwords))\
        .map(remove_ats)\
        .map(remove_links)\
        .str.join(" ")
    return dataframe
    
tmp_df = clean_strings(df.copy(), "tweet", "cleaned_tweet")
assert tmp_df is not None, "The cleaned DataFrame object was None. Did you forget to return a value from your function?"
assert "cleaned_tweet" in tmp_df, "The cleaned DataFrame doesn't contain the expected \"cleaned_tweet\" column. Did you make a new column with the cleaned strings?"
assert tmp_df.loc[4, "cleaned_tweet"] == "south america dominating here brazil much better * *", "Cleaned string doesn't match expected string. Check your cleaning function."
print("Cleaning tests succeeded!")
df = tmp_df

import sklearn.feature_extraction.text
import sklearn.metrics
import sklearn.model_selection
import sklearn.naive_bayes

def get_bag_of_words(dataframe, column):
    # Fill in the function to return a bag of words from the given dataframe
    count_vectorizer = sklearn.feature_extraction.text.CountVectorizer()
    bow = count_vectorizer.fit_transform(dataframe[column])
    return count_vectorizer, bow

bow_tuple = get_bag_of_words(df.copy(), "cleaned_tweet")
assert isinstance(bow_tuple, tuple), "The returned value is not a tuple as expected. Check your return statement (to return a tuple separate the elements you want to return with commas: `return \"string\", 123`)."
assert len(bow_tuple) == 2, f"The returned tuple doesn't have the expected length: {len(bow_tuple)} != 2"
_count_vectorizer, _bow = bow_tuple
assert isinstance(_count_vectorizer, sklearn.feature_extraction.text.CountVectorizer), "The first element of the returned tuple is not a CountVectorizer as expected."
assert hasattr(_bow, "shape"), "The second element of the returned tuple doesn't have the expected `shape` attribute. Is it a bag of words transformed with the count vectorizer?"
assert _bow.shape[0] == 50000, "Bag of words doesn't have the expected shape. Did you include all the text examples?"
assert _bow.shape[1] > 20000, "Your bag of words doesn't seem to include that many different words."
print("Bag of words tests succeeded!")

def categorize_sentiment(dataframe):
    # Fill in this function to use a classification model to predict sentiment.
    count_vectorizer, bow = get_bag_of_words(dataframe, "cleaned_tweet")
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(bow, df["sentiment"], test_size=0.1, random_state=np.random.RandomState(0))
    model = sklearn.naive_bayes.MultinomialNB()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(sklearn.metrics.classification_report(y_test, preds))
    return y_test, preds

classification_tuple = categorize_sentiment(df.copy())
assert isinstance(classification_tuple, tuple), "The returned value is not a tuple. Check your return statement."
assert len(classification_tuple) == 2, f"The returned tuple has an incorrect number of elements: {len(classification_tuple)} != 2"
y_test, preds = classification_tuple
assert hasattr(y_test, "shape"), "The returned value for true labels doesn't have a `shape` attribute. It probably doesn't have the expected type."
assert hasattr(preds, "shape"), "The returned value for predicted labels doesn't have a `shape` attribute. It probably doesn't have the expected type."
assert y_test.shape[0] == preds.shape[0], "Something's wrong: true labels and predicted labels don't have the same size."
assert sklearn.metrics.f1_score(y_test, preds, pos_label="positive") > 0.7, "Your F1 score is not as high as expected."
print("Sentiment classification tests passed!")


def predict_for_unknown_text(model, count_vectorizer):
    for sent in ["I'm very happy today", "I was pretty sad yesterday"]:
        print(sent, model.predict(count_vectorizer.transform([sent])))
        
def get_mispredicted_rows(dataframe, true_labels, predictions):
    # Fill in this function to find those data examples from the test set which your model misidentified
    p_compare = predictions == true_labels
    mask = p_compare[~p_compare]
    return dataframe.loc[mask.index]

mispredicted_rows = get_mispredicted_rows(df.copy(), y_test, preds)
assert isinstance(mispredicted_rows, pd.DataFrame), "The returned value is not a DataFrame as expected. Check your return statement."
print("Mispredicted rows tests passed!")
mispredicted_rows