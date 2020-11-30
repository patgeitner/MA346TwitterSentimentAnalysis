import re
import pandas as pd
import numpy as np
from nltk import PorterStemmer  # strips suffixes from a word
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding="ISO-8859-1", engine='python')
df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']  # naming the columns in the dataframe
df = df[['sentiment',
         'text']]  # the only relevant columns for this study are the tweet's content and corresponding sentiment
df = df.replace(to_replace=4, value=1)
training_df = df.reset_index(drop=True)


# function to remove a pattern patterns from the data
def remove_pattern(text, pattern):
    # re.findall() finds a pattern and puts it in a list
    r = re.findall(pattern, text)
    # re.sub() will remove the pattern from the sentences in the dataset
    for i in r:
        text = re.sub(i, "", text)
    return text


# removing twitter handles, numbers, special characters, and words shorter than 3 characters
training_df["cleaned_tweets"] = np.vectorize(remove_pattern)(training_df['text'], "@[\w]*")  # removing everything following an @ sign from the tweet
training_df['cleaned_tweets'] = training_df['cleaned_tweets'].str.replace("[^a-zA-Z#]", " ")  # replacing all characters that aren't letters or numbers with a space
training_df['cleaned_tweets'] = training_df['cleaned_tweets'].apply(lambda x: ' '.join([word for word in x.split() if len(word) > 3]))  # removes words shorter than 3 characters

tokenized_tweet = training_df['cleaned_tweets'].apply(lambda x: x.split())  # splitting tweets from dataset into lists of words
ps = PorterStemmer()  # bringing in NLTK's PorterStemmer function to removes suffixes
tokenized_tweet = tokenized_tweet.apply(lambda x: [ps.stem(i) for i in x])  # using PorterStemmer function on the lists of words
tokenized_tweet = tokenized_tweet.reset_index(drop=True)  # setting the index to sequential integers

# rejoining the lists of words now that the suffixes have been removed
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
training_df = training_df.reset_index(drop=True)  # resetting the index to sequential integers (indices were mixed up in sample)
training_df['cleaned_tweets'] = tokenized_tweet  # adding the tweets with suffixes removed to a column called cleaned tweets

tfidf = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')  # initializing a TfidfVectorizer function
tfidf_matrix = tfidf.fit_transform(training_df['cleaned_tweets'])  # transforming the cleaned tweets column to a tfidf vector
train_tfidf_matrix = tfidf_matrix[:1600000].todense()  # creating a dense matrix from the tfidf matrix of the cleaned tweets column
model = LogisticRegression(random_state=0, solver='lbfgs')
model.fit(train_tfidf_matrix, training_df['sentiment'])

filename1 = 'finalized_model.sav'
filename2 = 'cleaned_data.sav'
pickle.dump(model, open(filename1, 'wb'))
pickle.dump(training_df, open(filename2, 'wb'))
