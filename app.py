import re
import pandas as pd
import numpy as np
from nltk import PorterStemmer  # strips suffixes from a word
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st
import requests
import os


@st.cache(allow_output_mutation=True)
def read_data():
    df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding="ISO-8859-1", engine='python')
    df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
    df = df[['sentiment', 'text']]
    df = df.sample(500000)
    msk = np.random.rand(len(df)) < 0.8
    training_df = df[msk]
    test_df = df[~msk]
    return [training_df, test_df]


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
    dataframe["cleaned_tweets"] = np.vectorize(remove_pattern)(dataframe[column_name],
                                                               "@[\w]*")  # removing everything following an @ sign from the tweet
    dataframe['cleaned_tweets'] = dataframe['cleaned_tweets'].str.replace("[^a-zA-Z#]",
                                                                          " ")  # replacing all characters that aren't letters or numbers with a space
    dataframe['cleaned_tweets'] = dataframe['cleaned_tweets'].apply(
        lambda x: ' '.join([word for word in x.split() if len(word) > 3]))  # removes words shorter than 3 characters


@st.cache(allow_output_mutation=True)
def stemming():
    dfs = read_data()
    training_df = dfs[0]
    removing_unwanted_elements(training_df, 'text')
    tokenized_tweet = training_df['cleaned_tweets'].apply(lambda x: x.split())
    ps = PorterStemmer()
    tokenized_tweet = tokenized_tweet.apply(lambda x: [ps.stem(i) for i in x])
    tokenized_tweet = tokenized_tweet.reset_index(drop=True)
    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
    training_df = training_df.reset_index(drop=True)
    training_df['cleaned_tweets'] = tokenized_tweet
    return training_df


@st.cache(allow_output_mutation=True)
def binary():
    dfs = read_data()
    training_df = stemming()
    test_df = dfs[1]
    training_df = training_df.replace(to_replace=4, value=1)
    test_df = test_df.replace(to_replace=4, value=1)
    test_df = test_df.reset_index(drop=True)
    return [training_df, test_df]


@st.cache(allow_output_mutation=True)
def fit_training():
    tfidf = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(training_df['cleaned_tweets'])
    train_tfidf_matrix = tfidf_matrix[:500000].todense()
    x_train_tfidf, x_valid_tfidf, y_train_tfidf, y_valid_tfidf = train_test_split(train_tfidf_matrix,
                                                                                  training_df['sentiment'],
                                                                                  test_size=0.3, random_state=17)
    model = LogisticRegression(random_state=0, solver='lbfgs')
    model.fit(x_train_tfidf, y_train_tfidf)
    return [tfidf, model]


@st.cache(allow_output_mutation=True)
def get_negative_and_positive_tweets():
    negative_words = ' '.join(text for text in training_df['cleaned_tweets'][training_df['sentiment'] == 0])
    positive_words = ' '.join(text for text in training_df['cleaned_tweets'][training_df['sentiment'] == 1])
    return [negative_words, positive_words]


def create_url(id):
    tweet_fields = "tweet.fields=lang,author_id"
    ids = "ids=" + id
    url = "https://api.twitter.com/2/tweets?{}&{}".format(ids, tweet_fields)
    return url


def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers


def connect_to_endpoint(url, headers):
    response = requests.request("GET", url, headers=headers)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()


def individual_pred(text):
    tfidf_test_matrix = tfidf.transform([text])
    test_pred = model.predict_proba(tfidf_test_matrix)
    test_pred_int = test_pred[:, 1] >= 0.5
    test_pred_int = test_pred_int.astype(np.int)
    return test_pred_int[-1]


dfs_prepped = binary()
training_df = dfs_prepped[0]
test_df = dfs_prepped[1]

fit = fit_training()
tfidf = fit[0]
model = fit[1]

words_classified = get_negative_and_positive_tweets()
negative_words = words_classified[0]
positive_words = words_classified[1]

st.title("Natural Language Processing for Twitter Sentiment Analysis")
st.subheader('Made by Patrick Geitner')
st.write("GitHub Repository: https://github.com/patgeitner/MA346TwitterSentimentAnalysis")
st.header("Let's Analyze the Sentiment Behind Some Twitter Posts!")
st.sidebar.title("How do you want to access the tweet?")
input_type = st.sidebar.selectbox("Choose a method", ["Type or Paste a Tweet",
                                                      "Enter the URL to a Specific Tweet"])

if input_type == "Type or Paste a Tweet":
    user_text = st.text_input(label="Enter a Tweet to Analyze")
    if user_text == "":
        pass
    else:
        user_text = user_text.lower()
        if individual_pred(user_text) == 0:
            st.subheader("This is a Negative Tweet")
        else:
            st.subheader("This is a Positive Tweet")

        user_list = user_text.split(" ")
        for i in range(len(user_list)):
            if negative_words.count(" " + user_list[i]) > positive_words.count(" " + user_list[i]):
                st.markdown(f'<font color= red> {user_list[i]} </font>', unsafe_allow_html=True)

            elif positive_words.count(" " + user_list[i]) > negative_words.count(" " + user_list[i]):
                st.markdown(f'<font color= green>{user_list[i]}</font>', unsafe_allow_html=True)

            else:
                st.markdown(f'<font color= black>{user_list[i]}</font>', unsafe_allow_html=True)


elif input_type == "Enter the URL to a Specific Tweet":
    user_link = st.text_input(label="Enter the link of a Tweet to Analyze")
    if user_link == "":
        pass
    else:
        id = str(user_link[-19:])
        bearer_token = os.environ.get('bearer_token')
        url = create_url(id)
        headers = create_headers(bearer_token)
        json_response = connect_to_endpoint(url, headers)
        tweet = str(json_response['data'][0]['text'])
        st.write(tweet)
        tweet = tweet.lower()

        if individual_pred(tweet) == 0:
            st.subheader("This is a Negative Tweet")
        else:
            st.subheader("This is a Positive Tweet")

        user_list = tweet.split(" ")
        for i in range(len(user_list)):
            if negative_words.count(" " + user_list[i]) > positive_words.count(" " + user_list[i]):
                st.markdown(f'<font color= red> {user_list[i]} </font>', unsafe_allow_html=True)

            elif positive_words.count(" " + user_list[i]) > negative_words.count(" " + user_list[i]):
                st.markdown(f'<font color= green>{user_list[i]}</font>', unsafe_allow_html=True)

            else:
                st.markdown(f'<font color= black>{user_list[i]}</font>', unsafe_allow_html=True)

