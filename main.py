import numpy as np
import pandas as pd
import os
import helper_funcs as hf

device = "cpu" # change on Mac to "mps" for GPU support 
# if torch.backends.mps.is_available():
#     device = "mps"
# else:
#     device = "cpu"


def getData():
    """
    Load the quora question pairs dataset from data.
    """
    path = os.path.join(os.path.dirname(__file__), 'data', 'questions.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please ensure the file exists.")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("The dataset is empty. Please check the file content.")
    print(f"Loaded {len(df)} rows from the dataset.")
    #print the head of the dataframe
    print(df.head())
    return df

def splitData(df):
    """
    Split the dataframe into train, validation, and test sets.
    """
    # we will need to modify this in order to ensure we have enough matches in each set, but right now this is just random
    train_df = df.sample(frac=0.8, random_state=42)
    temp_df = df.drop(train_df.index)
    val_df = temp_df.sample(frac=0.5, random_state=42)
    test_df = temp_df.drop(val_df.index)

    print(f"Train set: {len(train_df)} rows")
    print(f"Validation set: {len(val_df)} rows")
    print(f"Test set: {len(test_df)} rows")

    return train_df, val_df, test_df

def tokenizeData(df, method):
    """
    Tokenize the questions in the dataframe using a simple whitespace tokenizer.
    """
    #not implemented error
    raise NotImplementedError("Tokenization method not implemented. Please implement the tokenizeData function.")


def calcJaccard(train_set):
    # we should build a helper function or use apis to calculate the jaccard similarity for each pair
    raise NotImplementedError("Jaccard similarity calculation not implemented. Please implement the calcJaccard function.")

def calcTFIDF(train_set):
    # build index and calculate the tf-idf for each pair
    raise NotImplementedError("TF-IDF calculation not implemented. Please implement the calcTFIDF function.")

def buildFeatureVector_for_Deep_Learning(train_set):
    raise NotImplementedError("Deep Learning feature vector construction not implemented. Please implement the buildFeatureVector_for_Deep_Learning function.")

#model options should be, 1) Jaccard, 2)TFIDF_and_CosineSim, 3) Levenshtein Distance, 4)Weighted Model 1, 5)Weighted Model 2 (no Levenshtein), 6) Deep Learning Model 
def evaluateModel(model, val_set):
    """
    Evaluate the model on the validation set and return the accuracy, precision, recall, F1-Score, and ROC AUC score.
    """
    # not implemented error
    raise NotImplementedError("Model evaluation not implemented. Please implement the evaluateModel function.")

if __name__ == "__main__":
    # Load the data
    df = getData()