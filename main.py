import numpy as np
import pandas as pd
import os
import helper_funcs as hf
import py_stringmatching as sm
import argparse
from gpt4 import openai_api_call

device = "cpu" # change on Mac to "mps" for GPU support 
# if torch.backends.mps.is_available():
#     device = "mps"
# else:
#     device = "cpu"

# overall goal: use methods like Jaccard, TF-IDF, and rule-based methods to build a feature vector for each pair of questions in the quora dataset, and then use a deep learning model to classify them as duplicates or not.
# we will compare to a simple approach of weighted scoring system of similarity scores from the different methods.
# if time allows we will compare against GPT-3 or other LLMs to see if they can classify the pairs as duplicates or not.

def main():
    parser = argparse.ArgumentParser(description="Question Duplicate Detection System")
    parser.add_argument(
        "--mode",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="""Select operation mode:
        1: Siamese Network
        2: GPT4 Analysis
        3: Classical Classifier
        4: Save Similarity Scores
        5: Miscellaneous Tests"""
    )
    args = parser.parse_args()
    
    if args.mode is None:
        print("\nPlease select a mode:")
        print("1: Siamese Network")
        print("2: GPT4 Analysis")
        print("3: Classical Classifier")
        print("4: Save Similarity Scores")
        print("5: Miscellaneous Tests")
        args.mode = int(input("\nEnter your choice (1-5): "))
        
    #load data first 
    df = getData()
    # TODO: split the data into train, validation, and test sets
    if args.mode == 1:
        print("Running Siamese Network...")
        # TODO: Add Siamese network logic
    elif args.mode == 2:
        print("Running GPT4 Analysis...")
        q1 = df.iloc[0]['question1']
        q2 = df.iloc[0]['question2']
        print(openai_api_call(q1, q2))
    elif args.mode == 3:
        print("Running Classical Classifier...")
        # TODO: Add classifier logic
    elif args.mode == 4:
        print("Saving Similarity Scores...")
        # TODO: Add similarity score logic
    elif args.mode == 5:
        print("Running Miscellaneous Tests...")
        # Current test logic here
        # for now just going to test some tokenization methods from https://anhaidgroup.github.io/py_stringmatching/v0.4.2/Tutorial.html
        # get an example q1 string and q2 string
        q1 = df.iloc[0]['question1']
        q2 = df.iloc[0]['question2']
        print(f"Example question 1: {q1}")
        print(f"Example question 2: {q2}")
        #TODO: clean the data, lowercase, do we want to remove special characters, do we want to remove stop words?
        # if we want to remove stop words we likely need to use a library like nltk
        # for now just going to lowercase the data
        q1 = q1.lower()
        q2 = q2.lower()
        # create a q3 tokenizer
        qg3_tok = sm.QgramTokenizer(qval=3)
        # tokenize the questions
        q1_tokens = qg3_tok.tokenize(q1)
        q2_tokens = qg3_tok.tokenize(q2)
        print(f"Tokenized question 1: {q1_tokens}")
        print(f"Tokenized question 2: {q2_tokens}")
        #lets also create a simple whitespace tokenizer (like a word based but we still have all sorts of potential characters)
        ws_tok = sm.WhitespaceTokenizer()
        q1_tokens_ws = ws_tok.tokenize(q1)
        q2_tokens_ws = ws_tok.tokenize(q2)
        print(f"Whitespace tokenized question 1: {q1_tokens_ws}")
        print(f"Whitespace tokenized question 2: {q2_tokens_ws}")
        # now lets calculate the jaccard similarity between the two tokenized questions
        #create a Jaccard similarity measure object
        jac = sm.Jaccard()
        test_jac_3gram = jac.get_raw_score(q1_tokens, q2_tokens) 
        # now lets calculate the jaccard similarity between the two whitespace tokenized questions
        test_jac_ws = jac.get_raw_score(q1_tokens_ws, q2_tokens_ws) 
        # create a Levenshtein similarity measure object
        lev = sm.Levenshtein()
        test_lev = lev.get_raw_score(q1, q2) # note q1 and q2 qre the og strings
        print(f"Jaccard similarity (3-gram): {test_jac_3gram}")
        print(f"Jaccard similarity (whitespace): {test_jac_ws}")
        print(f"Levenshtein similarity: {test_lev}")
    
    
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


# using py_stringmatching (see webpage: https://anhaidgroup.github.io/py_stringmatching/v0.4.2/Tutorial.html)
# Computing a similarity score between two given strings x and y then typically consists of four steps: 
# (1) selecting a similarity measure type
# (2) selecting a tokenizer type
# (3) creating a tokenizer object (of the selected type) and using it to tokenize the two given strings x and y
# (4) creating a similarity measure object (of the selected type) and applying it to the output of the tokenizer to compute a similarity score


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
    main()