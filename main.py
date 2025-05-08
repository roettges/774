import numpy as np
import pandas as pd
import os
import helper_funcs as hf
import py_stringmatching as sm
import argparse
import torch
from gpt4 import gpt4_analysis
#from siamese_model import train_siamese
import siamese_simple_main
from sklearn.model_selection import train_test_split
import preprocessData as preprocess
import miscDataReview as misc


device = "cpu" # change on Mac to "mps" for GPU support 
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# overall goal: use methods like Jaccard, TF-IDF, and rule-based methods to build a feature vector for each pair of questions in the quora dataset, and then use a deep learning model to classify them as duplicates or not.
# we will compare to a simple approach of weighted scoring system of similarity scores from the different methods.
# if time allows we will compare against GPT-3 or other LLMs to see if they can classify the pairs as duplicates or not.

def main():
    parser = argparse.ArgumentParser(description="Question Duplicate Detection System")
    parser.add_argument(
        "--mode",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help="""Select operation mode:
        1: Siamese Network
        2: GPT4 Analysis
        3: Run miscDataReview.py (only run if you have reviewed the script and files)
        4: Save Similarity Scores
        5: Calculate Distance Metrics for GPT4 embeddings
        6. Evaluate a Model""",
    )
    args = parser.parse_args()
    
    if args.mode is None:
        print("\nPlease select a mode:")
        print("1: Siamese Network")
        print("2: GPT4 Analysis")
        print("3: Run miscDataReview.py (only run if you have reviewed the script and files)")
        print("4: Save Similarity Scores")
        print("5: Distance Metrics for GPT4 embeddings")
        print("6: Evaluate a Model")
        args.mode = int(input("\nEnter your choice (1-6): "))
        
    

    if args.mode == 1:
        print("\nLaunching Siamese Network Training...")
        siamese_simple_main.main()
        return
        # print("Running Siamese Network...")
        # train_sample = train.sample(frac=0.01, random_state=42)
        # val_sample = val.sample(frac=0.01, random_state=42)
        # test_sample = test.sample(frac=0.01, random_state=42)

        # # jac = sm.Jaccard()
        # # lev = sm.Levenshtein()
        # # for d in [train_df, val_df, test_df]:
        # #     d['sim_jaccard'] = [jac.get_raw_score(a.lower().split(), b.lower().split()) for a, b in zip(d['question1'], d['question2'])]
        # #     d['sim_levenshtein'] = [lev.get_raw_score(a.lower(), b.lower()) for a, b in zip(d['question1'], d['question2'])]
    
        # train_siamese(train_sample, val_sample, test_sample, device=device, use_sim_features=False)
        
    elif args.mode == 2:
        print("Running GPT4 Analysis...")
        # TODO: combine plk files: https://stackoverflow.com/questions/76618091/how-to-merge-multiple-pickle-files-to-one-in-python
        
        results_1 = gpt4_analysis(val.iloc[:2000, :])
        hf.saveData(results_1, "gpt4o_results_2000")
        print("first 2000 done")

        results_2 = gpt4_analysis(val.iloc[2000:4000, :])
        hf.saveData(results_2, "gpt4o_results_4000")
        print("first 4000 done")

        results_3 = gpt4_analysis(val.iloc[4000:, :])
        hf.saveData(results_3, "gpt4o_results_end")
        print("DONE")
        return
    elif args.mode == 3:
        print("Execute miscDataReview.py File")
        print("miscDataReview.py relies on multiple hard coded data files. If you have not reviewed the script and files to confirm they exist in your directories then you should not procede.")
        print("Are you sure you want to procede? (y/n)")
        proceed = input("\nEnter your choice (y/n): ")
        if proceed.lower() != 'y':
            print("Exiting...")
            return
        else:
            print("Proceeding...")
            misc.main()
            return
    elif args.mode == 4:
        #load data first 
        df = getData()
        # preprocess the data
        df = preprocess.preprocessing(df)
        # df = pd.read_csv("data/questions.csv")
        train, test, val = splitData(df)
        print("Saving Similarity Scores...")
        #prompt user for which method to use
        print("Please select a method to use for similarity scoring:")
        print("1: Jaccard Similarity")
        print("2: TF-IDF and Cosine Similarity")
        print("3: Levenshtein Distance")
        
        method = int(input("\nEnter your choice (1-3): "))
        #get the similarity scores
        if method == 1:
            # make a copy of the dataframe
            df_copy = df.copy()
            #prompt for parsing method, 3-gram, space, lematized words
            print("Please select a parsing method, if you enter nothing or an invalid, then choice 1 will default:")
            print("1: 3-gram")
            print("2: Space")
            print("3: Lemmatized Words")
            # if they enter through without selecting, default to 3-gram
            parse_method = int(input("\nEnter your choice (1-3): "))
            # check if the user entered a valid choice and if null
            #if not 1, 2, or 3, default to 1
            if parse_method not in [1, 2, 3]:
                print("Invalid choice, defaulting to 3-gram, option 1")
                parse_method = 1
            #TODO: prompt for stop words removal
            print("Do you want to remove stop words?")
            print("1: Yes")
            print("2: No")
            # if they enter through without selecting, default to no
            stop_words = int(input("\nEnter your choice (1-2): "))
            if stop_words == 1:
                print("You chose to remove stop words")
                stop_words = True
            else:
                print("You chose not to remove stop words")
                stop_words = False
            #TODO: handle data cleaning, tokenization, and similarity score calculation
            hf.saveJaccard(df_copy, parse_method, stop_words)
        elif method == 2:
            #TODO: need to build index for the TF-IDF and Cosine Similarity
            #TODO: handle data cleaning, tokenization, and similarity score calculation
            #TODO: save the similarity scores to a file in the output folder
            pass
        elif method == 3:
            #TODO: handle data cleaning, tokenization, and similarity score calculation
            print("Do you want to remove stop words?")
            print("1: Yes")
            print("2: No")
            # if they enter through without selecting, default to no
            stop_words = int(input("\nEnter your choice (1-2): "))
            if stop_words == 1:
                print("You chose to remove stop words")
                stop_words = True
            else:
                print("You chose not to remove stop words")
                stop_words = False
            hf.saveLevenshtein(df, stop_words)
            #TODO: save the similarity scores to a file in the output folder
            pass
    elif args.mode == 5:
        print("Getting GPT4 embeddings...")
        embeddingsFilePath = "data/embeddings_and_data_with_embeddings/output_with_embeddings.csv"
        # check if the file exists
        if not os.path.exists(embeddingsFilePath):
            print(f"File {embeddingsFilePath} does not exist. Please check the path.")
            return
        # load the embeddings
        embedding_df = pd.read_csv(embeddingsFilePath)
        # check if the file is empty
        if embedding_df.empty:
            print(f"File {embeddingsFilePath} is empty. Please check the file.")
            return
        # add empty columns for cosine_sim, manhattan_dist, euclidean_dist
        embedding_df['cosine_sim'] = 0.0
        embedding_df['manhattan_dist'] = 0.0
        embedding_df['euclidean_dist'] = 0.0
        
        # for each row get & update the cosine similarity, manhattan distance, and euclidean distance
        for index, row in embedding_df.iterrows():
            # get the embeddings
            question1_embedding = row['question1_embedding']
            question2_embedding = row['question2_embedding']
            # convert to numpy array
            question1_embedding = np.fromstring(question1_embedding[1:-1], sep=',')
            question2_embedding = np.fromstring(question2_embedding[1:-1], sep=',')
            # calculate the cosine similarity
            cosine_sim = hf.cosine_similarity(question1_embedding, question2_embedding)
            # calculate the manhattan distance
            manhattan_dist = hf.manhattan_distance(question1_embedding, question2_embedding)
            # calculate the euclidean distance
            euclidean_dist = hf.euclidean_distance(question1_embedding, question2_embedding)
            
            # update the dataframe
            embedding_df.at[index, 'cosine_sim'] = cosine_sim
            embedding_df.at[index, 'manhattan_dist'] = manhattan_dist
            embedding_df.at[index, 'euclidean_dist'] = euclidean_dist
            
        # save the dataframe to a new file
        outputFilePath = "data/embeddings_and_data_with_embeddings/full_data_with_gpt4embeddings_and_distances.csv"
        embedding_df.to_csv(outputFilePath, index=False)
        # save a pickle file of the dataframe
        outputPicklePath = "data/embeddings_and_data_with_embeddings/full_data_with_gpt4embeddings_and_distances.pkl"
        embedding_df.to_pickle(outputPicklePath)
        # print the first 5 rows of the dataframe
        print(embedding_df.head())
        return
    elif args.mode == 6:
        print("Is this for gpt4 eval? If yes enter 1, otherwise enter anything else")
        gpt4_eval = int(input("\nEnter your choice (1-2): "))
        if gpt4_eval == 1:
            # read from /data/full_data_with_gpt4embeddings_and_distances.csv
            gpt4_df =  pd.read_csv("data/embeddings_and_data_with_embeddings/full_data_with_gpt4embeddings_and_distances.csv")
            train,test,val = splitData(gpt4_df)
            # prompt for threshold for cosine similarity
            print("Please enter the threshold for cosine similarity to be considered a duplicate:")
            thresh = input("\nEnter your choice (threshold): ")
            # if threshold is null or invalid, default to 0.5
            if thresh == "":
                thresh = 0.5
            else:
                # make sure threshold is a float
                try:
                    thresh = float(thresh)
                except ValueError:
                    print("Invalid threshold value. Please enter a number, try again.")
                    # quit the program
                    return
            hf.simpleEvaluateM(filePathName=None, gt_column="is_duplicate", pred_column="cosine_sim", threshold=thresh, data=val)
        else:
            # prompt user for file name
            print("Please enter the file name of the model to evaluate:")
            fname = input("\nEnter your choice (file name): ")
            print("Enter the column name of the ground truth labels, the default value is is_duplicate:")
            col_name = input("\nEnter your choice (column name): ")
            if col_name == "":
                col_name = "is_duplicate"
            print("Enter the column name of the predited labels, or if you only have a similarity score, enter the column name of the similarity score:")
            pred_col_name = input("\nEnter your choice (column name): ")
            if pred_col_name == "":
                pred_col_name = "similarity_score"
            print("If you specified a column for a similarity score, please enter the threshold for the similarity score to be considered a duplicate, otherwise you can leave this blank:")
            threshold = input("\nEnter your choice (threshold): ")
            if threshold == "":
                threshold = None
            else:
                #make sure threshold is a float
                try:
                    threshold = float(threshold)
                except ValueError:
                    print("Invalid threshold value. Please enter a number, try again.")
                    # quit the program
                    return
            # evaluate the model
            eval_results,fp,fn = hf.evaluateM(fname, col_name, pred_col_name, threshold)
        return
    
def getData():
    """
    Load the PREPROCESSED quora question pairs dataset from data.
    """
    path = os.path.join(os.path.dirname(__file__), 'data', 'questions.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please ensure the file exists.")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("The dataset is empty. Please check the file content.")
    print(f"Loaded {len(df)} rows from the dataset.")
    #print the head of the dataframe
    #print(df.head())
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
    min_class_size = small_train_size // 2 if small_train_size is not None else int(min(len(class_0), len(class_1)) * 0.9)

    # sample same number from both classes and combine into train df
    class_0_train = class_0.sample(n = min_class_size, random_state = 42)
    class_1_train = class_1.sample(n = min_class_size, random_state = 42)
    train_df = pd.concat([class_0_train, class_1_train])

    # remove the entries used in train df
    class_0_remaining = class_0.drop(class_0_train.index)
    class_1_remaining = class_1.drop(class_1_train.index)
    remaining_df = pd.concat([class_0_remaining, class_1_remaining])
    
    # split remaining data in half for test and val
    test_df, val_df = train_test_split(remaining_df, test_size = 0.5, random_state = 42)
    print(len(train_df), len(test_df), len(val_df))

    # sanity check to ensure no data leakage
    assert len(set(train_df.index).intersection(set(test_df.index))) == 0
    assert len(set(train_df.index).intersection(set(val_df.index))) == 0
    assert len(set(test_df.index).intersection(set(val_df.index))) == 0
    
    return train_df, test_df, val_df

if __name__ == "__main__":
    main()