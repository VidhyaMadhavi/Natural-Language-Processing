# https://pythonprogramminglanguage.com/kmeans-text-clustering/
# http://www.pitt.edu/~naraehan/presentation/Movie+Reviews+sentiment+analysis+with+Scikit-Learn.html
# https://www.linkedin.com/pulse/text-classification-using-bag-words-approach-nltk-scikit-rajendran/


# Importing Libraries
import import_data as id
import prepare_data as pre_data
import feature_engineering as fe
import modelling as mod



def split_reviews_to_words(rev):
    print (rev)


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
    split_reviews_to_words(reviews('review'))
    print("\n Data Loading and Preprocessing Completed...")



