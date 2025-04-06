import py_stringmatching as sm
import pandas as pd
import os
import numpy as np
import gzip
import pickle
# import spacy for lemmatization
import spacy
import time
from functools import wraps
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
# cpu_cores = os.cpu_count()
#global output directory
output_directory_csv = os.path.join(os.getcwd(), 'output')
output_directory_pickles = os.path.join(os.getcwd(), 'output_pickled')

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper


def evaluateM(filename, gt_column, pred_column, threshold=None):
    """
    Evaluate the model's predictions against the ground truth data using accuracy, precision, recall, F1-Score, and ROC AUC score.
    
    INPUTS:
    filename: the name of the file containing the ground truth data. It is assumed to be in pickle format in output_directory_pickled
    gt_column: the name of the column containing the ground truth data
    pred_column: the name of the column containing the model's predictions
    threshold: the threshold to use for binary classification. If None, the column is assumed to be binary already and will be used as is.
    
    """
    # Load the data from pickle
    df = loadData(filename, format='pickle')
    raise NotImplementedError("Model evaluation not implemented. Please implement the evaluateM function.")
@timer
def saveJaccard(data, parse_method, remove_stopwords):
    """
    Calculate the Jaccard similarity between the questions in the dataset.
    INPUTS:
    data: the dataset to calculate the Jaccard similarity on
    parse_method: the method to use for parsing the questions: 1: 3gram, 2: whitespace, 3: lemmatized words
    remove_stopwords: whether to remove stopwords from the questions. True if stopwords should be removed, False otherwise.
    OUTPUTS:
    None
    """
    cleaned_data = clean_data(data, remove_stopwords)
    jac = sm.Jaccard()
    start_total = time.time()
    if parse_method == 2:
        if remove_stopwords == True:
            filename = 'jaccard_ws_stopwordsremoved'
        else:
            filename = 'jaccard_ws'
        print("Calculating Jaccard similarity using whitespace tokenizer...")
        # create a whitespace tokenizer
        ws_tok = sm.WhitespaceTokenizer()
        # now lets calculate the jaccard similarity between the two whitespace tokenized questions
        cleaned_data['jaccard_ws'] = cleaned_data.apply(lambda x: jac.get_raw_score(ws_tok.tokenize(x['question1']), ws_tok.tokenize(x['question2'])), axis=1)
    elif parse_method == 3:
        if remove_stopwords == True:
            filename = 'jaccard_lemma_stopwordsremoved'
        else:
            filename = 'jaccard_lemma'
        print("Calculating Jaccard similarity using lemmatized words...")
        def lemmatized_tokens(text):
            doc = nlp(text)
            return [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
        
        def jaccard_score_lemmas(q1, q2):
            return jac.get_raw_score(lemmatized_tokens(q1), lemmatized_tokens(q2))
        
        cleaned_data['jaccard_lemma'] = cleaned_data.apply(lambda x: jaccard_score_lemmas(x['question1'], x['question2']), axis=1)
        #note to self - I do not expect this to do well namely with math questions, but it is a good test of the lemmatization process
    else: 
        if remove_stopwords == True:
            filename = 'jaccard_3gram_stopwordsremoved'
        else:
            filename = 'jaccard_3gram'
        print("Calculating Jaccard similarity using 3-gram...")
        # create a 3-gram tokenizer
        qg3_tok = sm.QgramTokenizer(qval=3)
        # now lets calculate the jaccard similarity between the two 3-gram tokenized questions
        cleaned_data['jaccard_3gram'] = cleaned_data.apply(lambda x: jac.get_raw_score(qg3_tok.tokenize(x['question1']), qg3_tok.tokenize(x['question2'])), axis=1) 
    # save the results to a file
    #savecsv(cleaned_data, filename)
    saveData(cleaned_data, filename, format='csv')
    saveData(cleaned_data, filename, format='pickle')
    end_total = time.time()
    print(f"\nTotal processing time for tokenization and jacc score calc {filename}: {end_total - start_total:.2f} seconds")
    return

def saveData(data, filename, format='pickle'):
    """
    Saves data in either CSV or compressed Pickle format.

    Parameters:
    - data (DataFrame): the pandas DataFrame to save
    - filename (str): file name without extension
    - format (str): 'csv' or 'pickle'
    """
        
    if format == 'csv':
        if not os.path.exists(output_directory_csv):
            os.makedirs(output_directory_csv)
        full_path = os.path.join(output_directory_csv, f"{filename}.csv")
        if os.path.exists(full_path):
            overwrite = input(f"{full_path} already exists. Do you want to overwrite it? (y/n): ")
            if overwrite.lower() != 'y':
                new_filename = input("Please enter a new filename (without extension): ")
                full_path = os.path.join(output_directory, f"{new_filename}.csv")
                data.to_csv(full_path, index=False)
                print(f"Saved CSV to {full_path}")
            else:
                data.to_csv(full_path, index=False)
                print(f"Overwrote CSV at {full_path}")
        else:
            data.to_csv(full_path, index=False)
            print(f"Saved CSV to {full_path}")
    elif format == 'pickle':
        if not os.path.exists(output_directory_pickles):
            os.makedirs(output_directory_pickles)
        # Save the data in compressed Pickle format
        full_path = os.path.join(output_directory_pickles, f"{filename}.pkl.gz")
        if os.path.exists(full_path):
            overwrite = input(f"{full_path} already exists. Do you want to overwrite it? (y/n): ")
            if overwrite.lower() != 'y':
                new_filename = input("Please enter a new filename (without extension): ")
                full_path = os.path.join(output_directory, f"{new_filename}.pkl.gz")
                with gzip.open(full_path, 'wb') as f:
                    pickle.dump(data, f)
                print(f"Saved Pickle to {full_path}")
            else:
                with gzip.open(full_path, 'wb') as f:
                    pickle.dump(data, f)
                print(f"Overwrote Pickle at {full_path}")
        else:
            # Save the data in compressed Pickle format
            with gzip.open(full_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved Pickle to {full_path}")
    else:
        raise ValueError("Invalid format. Use 'csv' or 'pickle'.")

def loadData(filename, format='pickle'):
    """
    Loads data in either CSV or compressed Pickle format.

    Parameters:
    - filename (str): file name without extension
    - format (str): 'csv' or 'pickle'

    Returns:
    - DataFrame
    """
    if format == 'csv':
        full_path = os.path.join(output_directory_csv, f"{filename}.csv")
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"{full_path} not found.")
        df = pd.read_csv(full_path)
    elif format == 'pickle':
        full_path = os.path.join(output_directory_pickles, f"{filename}.pkl.gz")
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"{full_path} not found.")
        with gzip.open(full_path, 'rb') as f:
            df = pickle.load(f)
    else:
        raise ValueError("Invalid format. Use 'csv' or 'pickle'.")

    print(f"Loaded data from {full_path}")
    return df

# def savecsv(d, filename):
#      #check if the file already exists, if so prompt the user to overwrite or not
#     if os.path.exists(filename):
#         overwrite = input(f"{filename} already exists. Do you want to overwrite it? (y/n): ")
#         if overwrite.lower() != 'y':
#             #prompt the user to enter a new filename
#             new_filename = input("Please enter a new filename (without extension): ")
#             filename = os.path.join(output_directory, f"{new_filename}.csv")
#             d.to_csv(filename, index=False)
#         #otherwise overwrite the file
#         else:
#             print(f"Overwriting {filename}...")
#             d.to_csv(filename, index=False)
#     #check if the output directory exists
#     else:
#         # create the output directory if it doesn't exist
#         if not os.path.exists(output_directory):
#             os.makedirs(output_directory)
#         # save the cleaned data to a file
#         print(f"Saving the results to {filename}...")
#         # save the cleaned data to a file`
#         d.to_csv(filename, index=False)
#     return
@timer
def clean_data(data, word_removal):
    """
    Clean the data by lowercasing the questions and removing stopwords.
    INPUTS:
    data: the dataset to clean
    remove_stopwords: whether to remove stopwords from the questions. True if stopwords should be removed, False otherwise.
    OUTPUTS:
    Dataframe with cleaned questions.
    """
    print("Cleaning the data...")
    # remove the row if either question1 or question2 is NaN
    print("Removing rows with NaN values...")
    data.dropna(subset=['question1', 'question2'], inplace=True)
    # for every data['question1'] check if it is a string, if not print error message and the data type and row number
    # for every data['question2'] check if it is a string, if not print error message and the data type and row number
    # if both are strings, continue
    print("Checking if all questions are strings...")
    for index, row in data.iterrows():
        if not isinstance(row['question1'], str):
            raise ValueError(f"Error: question1 is not a string. Data type: {type(row['question1'])}. Row: {index}")
        if not isinstance(row['question2'], str):
            raise ValueError(f"Error: question2 is not a string. Data type: {type(row['question2'])}. Row: {index}")
    print("Lowercasing the questions...")
    data['question1'] = data['question1'].str.lower()
    data['question2'] = data['question2'].str.lower()
    if word_removal == True:
        print("Removing stopwords from the questions...")
        stopwords = nlp.Defaults.stop_words
        keep_words = {'what', 'when', 'where', 'why', 'how', 'which', 'who', 'has', 'whenever', 'whatever', 'whereupon', 'whence', 'could', 'would', 'should', 'whereafter', 'during', 'wherever', 'whoever', 'through', 'sometime', 'next', 'whom', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten','eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fourty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'first'}
        stopwords = [word for word in stopwords if word not in keep_words]
        # remove all stopwords from the questions except for anything in keep_words
        data['question1'] = data['question1'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
        data['question2'] = data['question2'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
    return data
    
@timer
def saveLevenshtein(data, remove_stopwords):
    cleaned_data = clean_data(data, remove_stopwords)
    if remove_stopwords == True:
        filename = os.path.join(output_directory, 'levenshtein_stopwordsremoved')
    else:
        filename = os.path.join(output_directory, 'levenshtein')
    print("Calculating Levenshtein similarity...")
    lev = sm.Levenshtein()
    cleaned_data['levenshtein'] = cleaned_data.apply(lambda x: lev.get_raw_score(x['question1'], x['question2']), axis=1)
    # save the results to a file
    #savecsv(cleaned_data, filename)
    saveData(cleaned_data, filename, format='csv')
    saveData(cleaned_data, filename, format='pickle')
    print(f"Levenshtein similarity saved to {filename}")
    return

# def buildTFIDX_Index(df):
#     """
#     Build a TF-IDF index for the questions in the dataset.
#     INPUTS:
#     df: the dataset to build the index on
#     OUTPUTS:
#     tfidf_matrix: the TF-IDF matrix
#     """
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     # Create the TF-IDF vectorizer
#     tfidf_vectorizer = TfidfVectorizer()
#     # Fit and transform the questions
#     tfidf_matrix = tfidf_vectorizer.fit_transform(df['question1'].tolist() + df['question2'].tolist())
#     print("TF-IDF matrix shape:", tfidf_matrix.shape)
#     print("TF-IDF matrix size:", tfidf_matrix.data.nbytes / 1024**2, "MB")
#     # Save the TF-IDF matrix to a file
#     filename = os.path.join(output_directory, 'tfidf_matrix.npz')
#     from scipy import sparse
#     sparse.save_npz(filename, tfidf_matrix)
#     print(f"TF-IDF matrix saved to {filename}")
#     # Save the feature names to a file
#     feature_names = tfidf_vectorizer.get_feature_names_out()
#     feature_names_filename = os.path.join(output_directory, 'tfidf_feature_names.txt')
#     with open(feature_names_filename, 'w') as f:
#         for feature in feature_names:
#             f.write(f"{feature}\n")
#     print(f"TF-IDF feature names saved to {feature_names_filename}")
#     # Save the vectorizer to a file
#     vectorizer_filename = os.path.join(output_directory, 'tfidf_vectorizer.pkl')
#     import pickle
#     with open(vectorizer_filename, 'wb') as f:
#         pickle.dump(tfidf_vectorizer, f)
#     print(f"TF-IDF vectorizer saved to {vectorizer_filename}")
#     return tfidf_matrix