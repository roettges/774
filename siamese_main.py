import numpy as np
import pandas as pd
import os
# import helper_funcs as hf
# import py_stringmatching as sm
import argparse
import torch
# from gpt4 import gpt4_analysis
from siamese_model import train_siamese
from sklearn.model_selection import train_test_split
import sys
from datetime import datetime
import atexit


os.makedirs("logs", exist_ok=True)
log_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"logs/siamese_log_{log_time}.txt"
log_file = open(log_filename, 'w')
sys.stdout = log_file
sys.stderr = log_file
atexit.register(log_file.close)
print(f"Logging to {log_filename}")

def main():
    parser = argparse.ArgumentParser(description="Question Duplicate Detection System")
    parser.add_argument(
        "--mode",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help=("Select operation mode:\n"
              "1: Siamese Network\n"
              "2: GPT4 Analysis\n"
              "3: Classical Classifier\n"
              "4: Save Similarity Scores\n"
              "5: Miscellaneous Tests")
    )
    args = parser.parse_args()
    
    # If mode not provided in arguments, temporarily restore stdout for input:
    if args.mode is None:
        sys.__stdout__.write("\nPlease select a mode:\n")
        sys.__stdout__.write("1: Siamese Network\n")
        sys.__stdout__.write("2: GPT4 Analysis\n")
        sys.__stdout__.write("3: Classical Classifier\n")
        sys.__stdout__.write("4: Save Similarity Scores\n")
        sys.__stdout__.write("5: Miscellaneous Tests\n")
        sys.__stdout__.flush()
        # Temporarily switch stdout so the input prompt appears on the terminal
        original_stdout = sys.stdout
        sys.stdout = sys.__stdout__
        try:
            args.mode = int(input("\nEnter your choice (1-5): "))
        finally:
            sys.stdout = original_stdout

    print("Using device:", device)
    
    # Continue with the rest of your workflow...
    df = pd.read_csv("data/questions.csv")
    print(f"Loaded dataframe with {len(df)} rows")
    
    # # Create a fixed, shrunken validation set (small_val) from the full df.
    # # This uses the shrinkDataset and splitData functions below.
    # _, _, fixed_small_val = splitData(df, shrink_dataset=True)
    # print(f"Fixed small validation set size: {len(fixed_small_val)}")
    
    # # Remove the fixed small_val rows from df (using the id column, assumed to be the first column)
    # remaining_df = df[~df.iloc[:, 0].isin(fixed_small_val.iloc[:, 0])]
    
    # # Resample training and testing sets from the remaining data
    # train, test, val = splitData(remaining_df)
    # print(f"Train set size: {len(train)}")
    # print(f"Test set size: {len(test)}")
    # val = pd.concat([val, fixed_small_val]).drop_duplicates()
    # print(f"Validation set size: {len(val)}")
    
    train, test, val = splitData(df)
    
    if args.mode == 1:
        print("Running Siamese Network...")
        # TODO: Add Siamese network logic
        early_stop = train.sample(frac=0.05, random_state=42)
        train = train.drop(early_stop.index)
        
        train_sample = train.sample(frac=0.01, random_state=42)
        es_sample = early_stop.sample(frac=0.01, random_state=42)
        val_sample = val.sample(frac=0.01, random_state=42)
    
        # train_siamese(train_sample, es_sample, val_sample, device=device, use_sim_features=True)
        train_siamese(train, early_stop, val, device=device, use_sim_features=False)
        
    # elif args.mode == 2:
    #     print("Running GPT4 Analysis...")
    #     # TODO: added .head() to keep API costs low for now
    #     results = gpt4_analysis(df.head())
    #     print(results)
    #     # TODO: maybe change to pickle instead of csv later
    #     hf.saveData(results, "gpt4o_results", "csv")
    # elif args.mode == 3:
    #     print("Running Classical Classifier...")
    #     # TODO: Add classifier logic
    # elif args.mode == 4:
    #     print("Saving Similarity Scores...")
    #     #prompt user for which method to use
    #     print("Please select a method to use for similarity scoring:")
    #     print("1: Jaccard Similarity")
    #     print("2: TF-IDF and Cosine Similarity")
    #     print("3: Levenshtein Distance")
        
    #     method = int(input("\nEnter your choice (1-3): "))
    #     #get the similarity scores
    #     if method == 1:
    #         # make a copy of the dataframe
    #         df_copy = df.copy()
    #         #prompt for parsing method, 3-gram, space, lematized words
    #         print("Please select a parsing method, if you enter nothing or an invalid, then choice 1 will default:")
    #         print("1: 3-gram")
    #         print("2: Space")
    #         print("3: Lemmatized Words")
    #         # if they enter through without selecting, default to 3-gram
    #         parse_method = int(input("\nEnter your choice (1-3): "))
    #         # check if the user entered a valid choice and if null
    #         #if not 1, 2, or 3, default to 1
    #         if parse_method not in [1, 2, 3]:
    #             print("Invalid choice, defaulting to 3-gram, option 1")
    #             parse_method = 1
    #         #TODO: prompt for stop words removal
    #         print("Do you want to remove stop words?")
    #         print("1: Yes")
    #         print("2: No")
    #         # if they enter through without selecting, default to no
    #         stop_words = int(input("\nEnter your choice (1-2): "))
    #         if stop_words == 1:
    #             print("You chose to remove stop words")
    #             stop_words = True
    #         else:
    #             print("You chose not to remove stop words")
    #             stop_words = False
    #         #TODO: handle data cleaning, tokenization, and similarity score calculation
    #         hf.saveJaccard(df_copy, parse_method, stop_words)
    #     elif method == 2:
    #         #TODO: need to build index for the TF-IDF and Cosine Similarity
    #         #TODO: handle data cleaning, tokenization, and similarity score calculation
    #         #TODO: save the similarity scores to a file in the output folder
    #         pass
    #     elif method == 3:
    #         #TODO: handle data cleaning, tokenization, and similarity score calculation
    #         print("Do you want to remove stop words?")
    #         print("1: Yes")
    #         print("2: No")
    #         # if they enter through without selecting, default to no
    #         stop_words = int(input("\nEnter your choice (1-2): "))
    #         if stop_words == 1:
    #             print("You chose to remove stop words")
    #             stop_words = True
    #         else:
    #             print("You chose not to remove stop words")
    #             stop_words = False
    #         hf.saveLevenshtein(df, stop_words)
    #         #TODO: save the similarity scores to a file in the output folder
    #         pass
    # elif args.mode == 5:
    #     print("Running Miscellaneous Tests...")
    #     # Current test logic here
    #     # for now just going to test some tokenization methods from https://anhaidgroup.github.io/py_stringmatching/v0.4.2/Tutorial.html
    #     # get an example q1 string and q2 string
    #     q1 = df.iloc[0]['question1']
    #     q2 = df.iloc[0]['question2']
    #     print(f"Example question 1: {q1}")
    #     print(f"Example question 2: {q2}")
    #     #TODO: clean the data, lowercase, do we want to remove special characters, do we want to remove stop words?
    #     # if we want to remove stop words we likely need to use a library like nltk
    #     # for now just going to lowercase the data
    #     q1 = q1.lower()
    #     q2 = q2.lower()
    #     # create a q3 tokenizer
    #     qg3_tok = sm.QgramTokenizer(qval=3)
    #     # tokenize the questions
    #     q1_tokens = qg3_tok.tokenize(q1)
    #     q2_tokens = qg3_tok.tokenize(q2)
    #     print(f"Tokenized question 1: {q1_tokens}")
    #     print(f"Tokenized question 2: {q2_tokens}")
    #     #lets also create a simple whitespace tokenizer (like a word based but we still have all sorts of potential characters)
    #     ws_tok = sm.WhitespaceTokenizer()
    #     q1_tokens_ws = ws_tok.tokenize(q1)
    #     q2_tokens_ws = ws_tok.tokenize(q2)
    #     print(f"Whitespace tokenized question 1: {q1_tokens_ws}")
    #     print(f"Whitespace tokenized question 2: {q2_tokens_ws}")
    #     # now lets calculate the jaccard similarity between the two tokenized questions
    #     #create a Jaccard similarity measure object
    #     jac = sm.Jaccard()
    #     test_jac_3gram = jac.get_raw_score(q1_tokens, q2_tokens) 
    #     # now lets calculate the jaccard similarity between the two whitespace tokenized questions
    #     test_jac_ws = jac.get_raw_score(q1_tokens_ws, q2_tokens_ws) 
    #     # create a Levenshtein similarity measure object
    #     lev = sm.Levenshtein()
    #     test_lev = lev.get_raw_score(q1, q2) # note q1 and q2 qre the og strings
    #     print(f"Jaccard similarity (3-gram): {test_jac_3gram}")
    #     print(f"Jaccard similarity (whitespace): {test_jac_ws}")
    #     print(f"Levenshtein similarity: {test_lev}")
    
    
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

def shrinkDataset(df):
    # check distribution
    class_distribution = df['is_duplicate'].value_counts()
    print("Class distribution before sampling:\n", class_distribution)

    # stratified sampling to shrink dataset but maintain proportion
    # increase test_size to reduce the size of df_reduced
    df_reduced, _ = train_test_split(df, test_size=0.9, stratify=df['is_duplicate'], random_state=42)

    # check distribution after sampling
    reduced_class_distribution = df_reduced['is_duplicate'].value_counts()
    print("\nClass distribution after sampling:\n", reduced_class_distribution)
    
    return df_reduced

def splitData(df, small_train_size=None, shrink_dataset=False):
    """
    Split the dataframe into train, validation, and test sets.
    """
    # we will need to modify this in order to ensure we have enough matches in each set, but right now this is just random
    # train_df = df.sample(frac=0.8, random_state=42)
    # temp_df = df.drop(train_df.index)
    # val_df = temp_df.sample(frac=0.5, random_state=42)
    # test_df = temp_df.drop(val_df.index)

    # print(f"Train set: {len(train_df)} rows")
    # print(f"Validation set: {len(val_df)} rows")
    # print(f"Test set: {len(test_df)} rows")

    # return train_df, val_df, test_df

    ############## START OF NEW CODE ##############

    if shrink_dataset:
        df = shrinkDataset(df)
    class_0 = df[df['is_duplicate'] == 0]
    class_1 = df[df['is_duplicate'] == 1]

    # grab 80% of the rows of the smaller class, or small_train_size if provided
    # if shrink_dataset:
    #     min_class_size = small_train_size // 2 if small_train_size is not None else int(min(len(class_0), len(class_1)) * 0.9)
    # else:
    #     min_class_size = 134375
    min_class_size = small_train_size // 2 if small_train_size is not None else int(min(len(class_0), len(class_1)) * 0.9)

    # sample same number from both classes and combine into train df
    class_0_train = class_0.sample(n = min_class_size, random_state = 42)
    class_1_train = class_1.sample(n = min_class_size, random_state = 42)
    train_df = pd.concat([class_0_train, class_1_train])

    # remove the entries used in train df
    class_0_remaining = class_0.drop(class_0_train.index)
    class_1_remaining = class_1.drop(class_1_train.index)
    remaining_df = pd.concat([class_0_remaining, class_1_remaining])
    
    # # split remaining data in half for test and val
    # if shrink_dataset:
    #     test_df, val_df = train_test_split(remaining_df, test_size = 0.5, random_state = 42)
    # else:
    #     test_df, val_df = train_test_split(remaining_df, test_size = 0.47368421, random_state = 42)
    test_df, val_df = train_test_split(remaining_df, test_size = 0.5, random_state = 42)
    print(len(train_df), len(test_df), len(val_df))

    # sanity check to ensure no data leakage
    assert len(set(train_df.index).intersection(set(test_df.index))) == 0
    assert len(set(train_df.index).intersection(set(val_df.index))) == 0
    assert len(set(test_df.index).intersection(set(val_df.index))) == 0
    
    return train_df, test_df, val_df

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

def setup_logging():
    """Redirect stdout and stderr to a log file."""
    import atexit
    from datetime import datetime
    # Create a logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    log_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"logs/siamese_log_{log_time}.txt"
    log_file = open(log_filename, 'w')
    sys.stdout = log_file
    sys.stderr = log_file
    atexit.register(log_file.close)
    print(f"Logging to {log_filename}")

if __name__ == "__main__":
    # (set up devices, etc.)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")
    
    main()