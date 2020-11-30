import pickle
import streamlit as st
import requests
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# function to initialize the tfidf vectorizer and fit it to the dataset
@st.cache(allow_output_mutation=True)
def fit_tfidf():
    tfidf = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(training_df['cleaned_tweets'])
    return tfidf

# function to allow model to be applied to some text
def individual_pred(text):
    tfidf_test_matrix = tfidf.transform([text])  # transforming the text into a tfidf matrix
    test_pred = model.predict_proba(tfidf_test_matrix)  # using model to predict probability that the tweet is positive
    test_pred_int = test_pred[:, 1] >= 0.5  # if the probability is greater than .5 the prediction is true
    test_pred_int = test_pred_int.astype(np.int)  # changing the boolean prediction to a 1 or 0
    return test_pred_int[-1]  # returning the individual prediction associated with text entered


@st.cache(allow_output_mutation=True)
def get_negative_and_positive_tweets():
    # creating a string of all the words contained in positive tweets and all of the words contained in negative twets
    negative_words = ' '.join(text for text in training_df['cleaned_tweets'][training_df['sentiment'] == 0])
    positive_words = ' '.join(text for text in training_df['cleaned_tweets'][training_df['sentiment'] == 1])
    return [negative_words, positive_words]


# accessing tweets by their id using the twitter API
def create_url(id):
    tweet_fields = "tweet.fields=lang,author_id"
    ids = "ids=" + id
    url = "https://api.twitter.com/2/tweets?{}&{}".format(ids, tweet_fields)
    return url


# authorizing use of the twitter API using bearer token
def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers


# returning the json response associated with the tweet id
def connect_to_endpoint(url, headers):
    response = requests.request("GET", url, headers=headers)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()


# function to bring in the the model and training df which have been prepared externally and saved to pickle
@st.cache(allow_output_mutation=True)
def load_saved_files():
    model = pickle.load(open('finalized_model.sav', 'rb'))
    training_df = pickle.load(open('cleaned_data.sav', 'rb'))
    training_df = training_df.replace(to_replace=4, value=1)  # 4s were used to represent positives before but we will use 1s
    return [model, training_df]

saved_files = load_saved_files()
model = saved_files[0]
training_df = saved_files[1]

tfidf = fit_tfidf()

words_classified = get_negative_and_positive_tweets()
negative_words = words_classified[0]
positive_words = words_classified[1]

st.title("Natural Language Processing for Twitter Sentiment Analysis")
st.subheader('Made by Patrick Geitner')
st.write("GitHub Repository: https://github.com/patgeitner/MA346TwitterSentimentAnalysis")
st.header("Let's Analyze the Sentiment Behind Some Twitter Posts!")
st.sidebar.title("How do you want to access the tweet?")

# allowing the user to choose whether they want to enter text or the URL of a tweet from twitter
input_type = st.sidebar.selectbox("Choose a method", ["Type or Paste a Tweet",
                                                      "Enter the URL to a Specific Tweet"])
if input_type == "Type or Paste a Tweet":
    # asking the user to enter a tweet if they chose to type or paste a tweet
    user_text = st.text_input(label="Enter a Tweet to Analyze")
    if user_text == "":
        pass
    else:
        user_text = user_text.lower()  # converting text to lower case to make better predictions
        # calling the individual prediction function on the text and printing the model's prediction
        if individual_pred(user_text) == 0:
            st.subheader("This is a Negative Tweet")
        else:
            st.subheader("This is a Positive Tweet")

        user_list = user_text.split(" ")
        for i in range(len(user_list)):
            # displaying the word in red if it appears more times in the negative list of words
            if negative_words.count(" " + user_list[i]) > positive_words.count(" " + user_list[i]):
                st.markdown(f'<font color= red> {user_list[i]} </font>', unsafe_allow_html=True)

            # displaying the word in green if it appears more times in the positive list of words
            elif positive_words.count(" " + user_list[i]) > negative_words.count(" " + user_list[i]):
                st.markdown(f'<font color= green>{user_list[i]}</font>', unsafe_allow_html=True)

            # displaying the word in black if its not in the positive or negative list or appears an equal number of times
            else:
                st.markdown(f'<font color= black>{user_list[i]}</font>', unsafe_allow_html=True)


elif input_type == "Enter the URL to a Specific Tweet":
    # asking the user to enter a url if they chose to enter a link to a tweet
    user_link = st.text_input(label="Enter the link of a Tweet to Analyze")
    if user_link == "":
        pass
    else:
        id = str(user_link[-19:])  # extracting the tweet id from the link
        bearer_token = os.environ.get('bearer_token')  # accessing my personal bearer token for the twitter api
        url = create_url(id)  # creating a url to access the tweet contents using the api
        headers = create_headers(bearer_token)  # authorizing use of the api with my bearer token
        json_response = connect_to_endpoint(url, headers)  # retrieving the json response to the tweet entered
        tweet = str(json_response['data'][0]['text'])  # extracting the contents of the tweet from the json response
        st.write(tweet)  # displaying the tweet contents from the url entered
        tweet = tweet.lower()  # converting to lower case to make better predictions

        # calling individual prediction function on tweet contents from url entered and showing model prediction
        if individual_pred(tweet) == 0:
            st.subheader("This is a Negative Tweet")
        else:
            st.subheader("This is a Positive Tweet")

        user_list = tweet.split(" ")
        for i in range(len(user_list)):
            # displaying the word in red if it appears more in the list of negative words
            if negative_words.count(" " + user_list[i]) > positive_words.count(" " + user_list[i]):
                st.markdown(f'<font color= red> {user_list[i]} </font>', unsafe_allow_html=True)

            # displaying the word in green if it appears more in the positive list of words
            elif positive_words.count(" " + user_list[i]) > negative_words.count(" " + user_list[i]):
                st.markdown(f'<font color= green>{user_list[i]}</font>', unsafe_allow_html=True)

            # displaying the word in black if it does not appear in the list of positive or negative words or shows up an equal number of times in each
            else:
                st.markdown(f'<font color= black>{user_list[i]}</font>', unsafe_allow_html=True)
