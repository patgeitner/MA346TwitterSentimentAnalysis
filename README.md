## Twitter Sentiment Analysis
This project performs an analysis of data about Twitter posts. The study includes descriptive analytics about what people are talking positively and negatively about and also trains a logistic model on a dataset of 1.6 million Twitter posts to predict whether some text is positive or negative. The app that uses the model can be found at: [Project Dashboard](https://nlp-twitter-sentiment.herokuapp.com/)

You can also run the code for creating the logistic model and some data visualization from this link: [Project Deepnote File](https://deepnote.com/project/fa91b211-2975-412c-a8b0-839a93b97fa6). Unfortunately, due to Deepnote's limited free computing power, the model in this notebook is only trained on 50,000 entries. However, the model created in model.py and used in app.py is trained on all 1.6 million entries from the dataset.

A formal report aboout this project can be found at here: [Report PDF](https://github.com/patgeitner/MA346TwitterSentimentAnalysis/blob/main/Final%20Project%20Report.pdf)


## Motivation
I wanted to build something that could classify text by drawing some meaning from words and came across this dataset on Kaggle found here: [Kaggle Dataset](https://www.kaggle.com/kazanova/sentiment140). I also wanted to analyze the things that people were praising and complaining about over social media to see if I could find meaningful patterns that went beyond expressive words like "love", "hate", and "like". Additionally, I built this app to gain experience with some Machine Learning and Natural Language Processing techniques as well as some web scraping.

## File Information
Various files that were necessary to deploy the app using Heroku are located in the HerokuFiles folder. The app.py script contains the code used to deploy the dashboard through Heroku. The model.py script contains the code to build and save the final model trained on all 1.6 million rows in the dataset. The final cleaned datset is contained in the cleaned_data.sav file. The visualizationAndModelFitting.ipynb file is a Jupyter notebook that walks through the process of fitting the logistic model to the dataset as well as creating data visualization for this project. A formal final report for this project can be found in the Final Project Report.docx file.

## Features
The app created in this project allows a user to enter some text, or the URL to a specific Twitter post. The model is then applied to the text or Tweet, and classifies it as either having a positive or negative sentiment. It also displays all of the words from the text entered in green if it is a "positive word", red if it is a "negative word", and black if the word is not seen in the dataset or seen the same amount in positive and negative tweets.

## Tech Used
This project was written in Python. The data for this project was prepared using the Pandas, Natural Language Tool Kit, NumPy, and Re libraries. SkLearn was used to fit the model and web scraping from Twitter URLs was done using the Twitter API. The Dashboard was created using the Streamlit library and deployed using Heroku. 

## Screenshots
![](/images/Screenshot.png)

![](/images/Screenshot2.png)

