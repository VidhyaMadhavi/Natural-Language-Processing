#--------------------------------------------------------------------------------------
#--------------------------------------------------
#Program Name: import_data.py
#Description: This file is used to load the required training and testing movie review data.
# the reviews need to be in .txt files. For the given directory all the text files will be
# loaded in this program.
#----------------------------------------------------
#--------------------------------------------------------------------------------------
import os
import pandas as pd


# --------------------------------------------------
# Function Name: merge_files(path)
# Description : Open each file and merge the content of all the files.
# --------------------------------------------------
def merge_files(path, list_file_names):
    merged_content = []
    for file in list_file_names:
        with open(path+file , encoding="latin1") as f:
            merged_content.append(f.read())
    return merged_content



#--------------------------------------------------
#Function Name: import_files(path)
#Description : From the given path all the text files are listed out
#--------------------------------------------------
def import_files(path ):
    data = [x for x in os.listdir(path) if x.endswith(".txt")]
    merged_content = merge_files(path,data )
    return (data, merged_content)

#--------------------------------------------------
#Function Name: create_dataframe(train_pos_merge_content,
                     # train_neg_merge_content,
                     # test_pos_merge_content,
                     # test_neg_merge_content)
#Description : Create a data frame with all the reviews
#--------------------------------------------------
def create_dataframe(train_pos_merge_content,train_neg_merge_content,
                     test_pos_merge_content,test_neg_merge_content,
                     train_positiveFiles,train_negativeFiles,
                     test_positiveFiles,test_negativeFiles):
    reviews = pd.concat([
        pd.DataFrame({"review": train_pos_merge_content, "label": 1, "file": train_positiveFiles}),
        pd.DataFrame({"review": train_neg_merge_content, "label": 0, "file": train_negativeFiles}),
        pd.DataFrame({"review": test_pos_merge_content, "label": 1, "file": test_positiveFiles}),
        pd.DataFrame({"review": test_neg_merge_content, "label": 0, "file": test_negativeFiles})
    ], ignore_index=True).sample(frac=1, random_state=1)

    print (reviews.head())
    return reviews