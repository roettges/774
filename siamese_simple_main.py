import numpy as np
import pandas as pd
import os
import argparse
import torch
from siamese_model import train_siamese
from sklearn.model_selection import train_test_split
import sys
from datetime import datetime
import atexit
import preprocessData as pp
from siamese_gpt_model import train_siamese_gpt
from siamese_model_with_gpt_scores import train_siamese_BERT_gpt_combo

def setup_logging():
    os.makedirs("logs", exist_ok=True)
    log_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"logs/siamese_log_{log_time}.txt"
    log_file = open(log_filename, 'w')
    sys.stdout = log_file
    sys.stderr = log_file
    atexit.register(log_file.close)
    print(f"Logging to {log_filename}")

def main():
    # Check if any args were passed to trigger interactive prompt
    no_args_passed = len(sys.argv) == 1

    parser = argparse.ArgumentParser(description='Run Siamese Network Training')
    parser.add_argument('--mode', type=int, choices=[1, 2, 3],
                        help='1: BERT, 2: GPT, 3: BERT+GPT')
    args = parser.parse_args()

    if not no_args_passed:
        setup_logging()  # Only log if NOT running interactively

    if args.mode is None and no_args_passed:
        while True:
            print("\nPlease select a mode:")
            print("1: Train BERT Siamese Network")
            print("2: Train BERT+GPT Combined Network")
            print("3: Train Siamese Network with GPT Encodings")
            try:
                mode = int(input("\nEnter mode (1-3): "))
                if mode in [1, 2, 3]:
                    args.mode = mode
                    break
                else:
                    print("Invalid mode. Please select 1, 2, or 3.")
            except ValueError:
                print("Invalid input. Please enter a number.")


    # Set device inside main
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:2")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    if args.mode == 1:
        df = pd.read_csv("data/preprocessedquestions.csv")
        train, test, val = splitData(df)
        early_stop = train.sample(frac=0.05, random_state=42)
        train = train.drop(early_stop.index)
        train_siamese(train, early_stop, val, device=device)

    elif args.mode == 2:
        df = pd.read_csv("data/full_data_with_gpt4embeddings_and_distances.csv")
        train, test, val = splitData(df)
        early_stop = train.sample(frac=0.05, random_state=42)
        train = train.drop(early_stop.index)
        train_siamese_BERT_gpt_combo(train, early_stop, test, device=device)

    elif args.mode == 3:
        df = pd.read_csv("data/output_with_embeddings.csv")
        train, test, val = splitData(df)
        early_stop = train.sample(frac=0.05, random_state=42)
        train = train.drop(early_stop.index)
        train_siamese_gpt(train, early_stop, val, device=device)

    else:
        print("Invalid mode selected. Exiting.")
        sys.exit(1)

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
    main()