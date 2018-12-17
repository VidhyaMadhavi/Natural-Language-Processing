#--------------------------------------------------------------------------------------
#--------------------------------------------------
#Program Name: modelling.py
#Description: This modules helps in designing
#different models
#----------------------------------------------------
#--------------------------------------------------------------------------------------
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from nltk.classify.scikitlearn import SklearnClassifier
#kmeans - text clustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def model_naive_bayes(X_train, X_test, y_train, y_test):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    print ("The training accuracy of MultinomialNB()is :", model.score(X_train, y_train))
    print ("The testing accuracy of MultinomialNB()is :", model.score(X_test, y_test))



def model_classifier_NB(X_train, X_test, y_train, y_test):
    print ("type of object:", type(X_train))
    # converting sparse matrix to DataFrame, use .todense() or .toarray()
    print("----------------Training Data: Converting sparse matrix to DataFrame-------------------------")
    training_set = pd.DataFrame(X_train.toarray())
    training_set['labels'] = y_train
    print (training_set.head(2))

    testing_set = pd.DataFrame(X_test.toarray())
    model = SklearnClassifier(MultinomialNB())
    model.train(training_set)
    print("The training accuracy of NLTK Classifier MultinomialNB()is :", model.classify.accuracy(model,training_set))
    print("The testing accuracy of NLTK Classifier MultinomialNB()is :", model.classify.accuracy(model, testing_set))

#--------------------------------------------------------------------------------------
# Text clustering
# After we have numerical features, we initialize the KMeans algorithm with K=2.
# print the top words per cluster.
# we give a new document to the clustering algorithm and let it predict its class.
#--------------------------------------------------------------------------------------
# def model_kmeans_text_clustering(X_train, X_test, y_train, y_test, vectorizer):
#     true_k = 2
#     model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
#     model.fit(X_train)
#
#     print("Top terms per cluster:")
#     order_centroids = model.cluster_centers_.argsort()[:, ::-1]
#     terms = vectorizer.get_feature_names()
#
#     for i in range(true_k):
#         print("Cluster %d:" % i),
#         for ind in order_centroids[i, :10]:
#             print(' %s' % terms[ind]),
#         print
#
#     print("\n")
#     print("Prediction: Review:- Movie is good")
#
#     Y = vectorizer.transform(["Movie is good."])
#     prediction = model.predict(Y)
#     print(prediction)
#
#     print("\n")
#     print("Prediction: Review:- Movie is bad")
#     Y = vectorizer.transform(["Movie is bad."])
#     prediction = model.predict(Y)
#     print(prediction)
#
#     print("The Accuracy of Kmeans for Training Dataset:", model.score(X_train,y_train))
#     print("The Accuracy of Kmeans for Testing Dataset:", model.score(X_test, y_test))
#
