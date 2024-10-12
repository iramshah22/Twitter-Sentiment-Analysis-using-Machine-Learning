# Twitter-Sentiment-Analysis-using-Machine-Learning

# Overview
This project aims to perform sentiment analysis on a large dataset of tweets, determining whether the sentiment expressed is positive, negative, or neutral. Sentiment analysis is a key task in natural language processing (NLP), widely used in social media monitoring, customer feedback analysis, and more. The dataset for this project is sourced from Kaggle's Sentiment140 dataset, containing 1.6 million labeled tweets.

# Purpose
The main goal of this project is to classify tweets based on their sentiment. By analyzing patterns in the text data, the model can predict the sentiment of new tweets and offer insights into public opinion or brand sentiment. This project demonstrates the use of the logistic regression algorithm for text classification, as well as preprocessing steps to handle raw text data.

# Dataset
The Sentiment140 dataset used in this project contains tweets with labels for positive (1) and negative (0) sentiments. The dataset was obtained from Kaggle using the Kaggle API. The dataset includes various features such as:
Tweet text
Polarity (sentiment label)
Tweet date
User information

# Process

1. Data Collection:

The Kaggle API is used to download the Sentiment140 dataset directly.
The dataset is extracted from a compressed file format using Python's ZipFile library.

2. Data Preprocessing:

Tweets undergo various preprocessing steps, including cleaning text by removing unwanted characters, numbers, and symbols using regular expressions (re library).
Stopwords (commonly used words like "the", "is", "in") are removed using NLTK's stopwords corpus to ensure only meaningful words are processed.
Stemming is applied using the PorterStemmer to reduce words to their base form (e.g., "running" becomes "run").

3. Feature Extraction:

The TfidfVectorizer is used to convert textual data into numerical form using the TF-IDF (Term Frequency-Inverse Document Frequency) method. This method helps in quantifying the importance of words in the corpus while reducing the impact of common but less important words.


# Model Building:
A Logistic Regression model is trained to classify the tweets into positive or negative sentiment.
The dataset is split into training and testing sets using train_test_split to evaluate model performance on unseen data.


# Evaluation:
The model’s accuracy is measured using accuracy_score to determine how well it performs on the test data.


# Technologies Used
Python
Kaggle API
NumPy and Pandas for data manipulation
NLTK for text preprocessing (stopwords, stemming)
Scikit-learn for feature extraction, model building, and evaluation


# Conclusion
This project demonstrates the application of machine learning for sentiment analysis on Twitter data. By employing logistic regression and thorough data preprocessing, we can accurately predict the sentiment of tweets, providing valuable insights into social media trends and public opinion. Future improvements may involve experimenting with other models like SVM, Naive Bayes, or deep learning methods to enhance performance.

