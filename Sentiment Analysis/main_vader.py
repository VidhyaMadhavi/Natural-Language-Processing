#--------------------------------------------------------------------------------------
#--------------------------------------------------
#Project Name: Project_Perform_Sentiment_Analsis_on_Movie_Reviews
#File Name: main.py
#Description: The project performs Sentiment Analysis on Movie Reviews
# if the movie review is positive, negative or neutral
#----------------------------------------------------
#--------------------------------------------------------------------------------------
# 1  Data Import/Data Load
# 2  Data Preprocessing
# 3  Feature Engineering
# 4  Feature Selection
# 5  Model Architecture
# 6  Model Training
# 7  Model Evaluation
# 7.1  Accuracy & Loss
# 7.2  Error Analysis
# 8  Model Application
# 8.1  Test Predictions
# 8.2  Custom Reviews
#--------------------------------------------------------------------------------------
# Importing Libraries
import import_data as id
import prepare_data as pre_data
import feature_engineering as fe
import modelling as mod
from sklearn import metrics
#Download the following
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk import sentiment
from nltk import word_tokenize
# import the relevant modules from the NLTK library
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def use_vader(List_of_reviews):
    sid = SentimentIntensityAnalyzer()
    pred_labels = []
    for sentence in List_of_reviews:

        scores = sid.polarity_scores(sentence)
        if scores['pos'] > scores['neg']:
            pred_labels.append(1)
        else:
            pred_labels.append(0)


    acc = metrics.accuracy_score(reviews['label'], pred_labels)
    print ("\n\n The Accuracy of the Vader tool is :", acc)



if __name__ == "__main__":

    print ("\n\n=================== Sentiment Analysis on Movie Reviews ===================")
    print ("\n\n------------------ Data Loading and Preprocessing ------------------")
    # Data Import/Data Load - Import text files
    # Load the Training Data
    path = './data_movie_reviews/train/pos/'
    train_positiveFiles, train_pos_merge_content = id.import_files(path)
    print ("Total of {} Files is loaded with Positive Sentiment for Training".format(len(train_positiveFiles)))

    path = './data_movie_reviews/train/neg/'
    train_negativeFiles, train_neg_merge_content = id.import_files(path)
    print("Total of {} Files is loaded with Negative Sentiment for Training".format(len(train_negativeFiles)))

    # Load the Testing Data
    path = './data_movie_reviews/test/pos/'
    test_positiveFiles, test_pos_merge_content = id.import_files(path)
    print("Total of {} Files is loaded with Positive Sentiment for Testing".format(len(test_positiveFiles)))

    path = './data_movie_reviews/test/neg/'
    test_negativeFiles, test_neg_merge_content = id.import_files(path)
    print("Total of {} Files is loaded with Negative Sentiment for Testing".format(len(test_negativeFiles)))

    print ("\n\n---------------------Dataframe of all the training and testing reviews---------------------\n ")
    reviews = id.create_dataframe(train_pos_merge_content,train_neg_merge_content,
                            test_pos_merge_content,test_neg_merge_content,
                            train_positiveFiles,train_negativeFiles,
                            test_positiveFiles,test_negativeFiles)

    print("\n Data Loading and Preprocessing Completed...")
    List_of_reviews = list(reviews["review"].values)
    use_vader(List_of_reviews)

    print ("\n\n*****************End of the Project*****************")
