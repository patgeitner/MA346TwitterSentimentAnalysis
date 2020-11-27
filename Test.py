import re
import pandas as pd
import numpy as np
from nltk import PorterStemmer  # strips suffixes from a word
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st


@st.cache(allow_output_mutation=True)
def read_data():
    df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding="ISO-8859-1", engine='python')
    df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
    df = df[['sentiment', 'text']]
    return df


@st.cache(allow_output_mutation=True)
def remove_pattern(text, pattern):
    # re.findall() finds a pattern and puts it in a list
    r = re.findall(pattern, text)
    # re.sub() removes @user from the sentences in the dataset
    for i in r:
        text = re.sub(i, "", text)
    return text


@st.cache(allow_output_mutation=True)
def removing_unwanted_elements(dataframe, column_name):
    dataframe["cleaned_tweets"] = np.vectorize(remove_pattern)(dataframe[column_name], "@[\w]*")  # removing everything following an @ sign from the tweet
    dataframe['cleaned_tweets'] = dataframe['cleaned_tweets'].str.replace("[^a-zA-Z#]"," ")  # replacing all characters that aren't letters or numbers with a space
    dataframe['cleaned_tweets'] = dataframe['cleaned_tweets'].apply(lambda x: ' '.join([word for word in x.split() if len(word) > 3]))  # removes words shorter than 3 characters


@st.cache(allow_output_mutation=True)
def stemming():
    df = read_data()
    removing_unwanted_elements(df, 'text')
    tokenized_tweet = df['cleaned_tweets'].apply(lambda x: x.split())
    ps = PorterStemmer()
    tokenized_tweet = tokenized_tweet.apply(lambda x: [ps.stem(i) for i in x])
    tokenized_tweet = tokenized_tweet.reset_index(drop=True)
    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
    df = df.reset_index(drop=True)
    df['cleaned_tweets'] = tokenized_tweet
    return df


@st.cache(allow_output_mutation=True)
def binary():
    df = stemming()
    df = df.replace(to_replace=4, value=1)
    df = df.reset_index(drop=True)
    return df


@st.cache(allow_output_mutation=True)
def fit_training():
    tfidf = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['cleaned_tweets'])
    train_tfidf_matrix = tfidf_matrix.todense()
    x_train_tfidf, x_valid_tfidf, y_train_tfidf, y_valid_tfidf = train_test_split(train_tfidf_matrix,
                                                                                  df['sentiment'],
                                                                                  test_size=0.3, random_state=17)
    model = LogisticRegression(random_state=0, solver='lbfgs')
    model.fit(x_train_tfidf, y_train_tfidf)
    return [tfidf, model]


def individual_pred(text):
    tfidf_test_matrix = tfidf.transform([text])
    test_pred = model.predict_proba(tfidf_test_matrix)
    test_pred_int = test_pred[:, 1] >= 0.5
    test_pred_int = test_pred_int.astype(np.int)
    return test_pred_int[-1]


df = binary()


fit = fit_training()
tfidf = fit[0]
model = fit[1]
print('done')

st.title("Natural Language Processing for Twitter Sentiment Analysis")
st.header("Let's Analyze the Sentiment Behind Some Twitter Posts!")


st.sidebar.title("How do you want to access the tweet?")
input_type = st.sidebar.selectbox("Choose a method", ["Type or Paste a Tweet",
                                                      "Enter the URL to a Specific Tweet",
                                                      "Enter a Username"])

if input_type == "Type or Paste a Tweet":
    user_text = st.text_input(label="Enter a Tweet to Analyze")
    if individual_pred(user_text) == 0:
        st.write("This is a Negative Tweet")
    else:
        st.write("This is a Positive Tweet")

elif input_type == "Enter the URL to a Specific Tweet":
    user_link = st.text_input(label="Enter the link of a Tweet to Analyze")
    if individual_pred(user_link) == 0:
        st.write("This is a Negative Tweet")
    else:
        st.write("This is a Positive Tweet")

elif input_type == "Enter a Username":
    user_handle = st.text_input(label="Enter Twitter Handle to Analyze their Last Tweet")
    if individual_pred(user_handle) == 0:
        st.write("This is a Negative Tweet")
    else:
        st.write("This is a Positive Tweet")
