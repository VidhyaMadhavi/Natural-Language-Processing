#--------------------------------------------------------------------------------------
#--------------------------------------------------
#Program Name: hyperparameter_fine_tuning.py
#Description: This module finds out the
# best hyperparameters for different models
#----------------------------------------------------
#--------------------------------------------------------------------------------------
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pandas as pd

def NB_parameter_tuning(X_train, X_test, y_train, y_test):
    list_alpha = np.arange(1 / 100000, 20, 0.11)
    score_train = np.zeros(len(list_alpha))
    score_test = np.zeros(len(list_alpha))
    recall_test = np.zeros(len(list_alpha))
    precision_test = np.zeros(len(list_alpha))
    count = 0
    for alpha in list_alpha:
        bayes = MultinomialNB(alpha=alpha)
        bayes.fit(X_train, y_train)
        score_train[count] = bayes.score(X_train, y_train)
        score_test[count] = bayes.score(X_test, y_test)
        recall_test[count] = metrics.recall_score(y_test, bayes.predict(X_test))
        precision_test[count] = metrics.precision_score(y_test, bayes.predict(X_test))
        count = count + 1

    matrix = np.matrix(np.c_[list_alpha, score_train, score_test, recall_test, precision_test])
    models = pd.DataFrame(data=matrix, columns=
    ['alpha', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
    models.head(n=20)

    #Lets select the model with the most test precision
    best_index = models['Test Precision'].idxmax()
    print (models.iloc[best_index, :] )