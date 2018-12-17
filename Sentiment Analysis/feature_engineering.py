#--------------------------------------------------------------------------------------
#--------------------------------------------------
#Program Name: feature_engineering.py
#Description: This modules helps in identifying the
#important features from the data
#Text preprocessing,
# tokenizing
# filtering of stopwords
# are included in a high level component that is able
# to build a dictionary of features and transform
# documents to feature vectors.
# We remove the stop words in order to improve the analytics
#----------------------------------------------------
#--------------------------------------------------------------------------------------

from sklearn import feature_extraction
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer # create bag of words
#kmeans - text clustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

def bag_of_words(reviews, number_of_features = 5000):
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=5000)
    f = feature_extraction.text.CountVectorizer(stop_words='english', ngram_range=(2, 2), max_features=number_of_features)
    X = f.fit_transform(reviews["review"])
    print ("The bag of words data frame has the following shape:",np.shape(X))
    print("It means that bag of words has {} features and has {} reviews in the dataframe".format(np.shape(X)[1], np.shape(X)[0]))
    return X

#--------------------------------------------------------------------------------------
# Function Name : tdidf():
# Feature extraction
# KMeans normally works with numbers only: we need to have numbers.
# To get numbers, we do a common step known as feature extraction.
#
# The feature we’ll use is TF-IDF, a numerical statistic.
# This statistic uses term frequency and inverse document frequency.
#
# we use statistics to get to numerical features.
# We’ll use the existing implementation of the TF-IDF algorithm in sklearn.
#
# The method TfidfVectorizer() implements the TF-IDF algorithm.
# TfidfVectorizer converts a collection of raw documents to a matrix of TF-IDF features.
#--------------------------------------------------------------------------------------
def tfidf(reviews):
    punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', "%"]
    stop_words = text.ENGLISH_STOP_WORDS.union(punc)
    vectorizer = TfidfVectorizer(stop_words=stop_words , max_features=5000)
    # vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(reviews["review"])
    return (X,vectorizer )


