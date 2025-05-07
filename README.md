# 774
Repository for our 774 project for question matching

# Install requirements
pip install numpy pandas py_stringmatching torch scikit-learn openai sentence-transformers tqdm num2words

pip install -U sentence-transformers

*** You will also need to install spacy for lemmatization and stopword removal ***
conda install -c conda-forge spacy
*** or if you don't want to use conda for installing spacy use the following***
pip install -U pip setuptools wheel
pip install -U spacy

*** Download English model for spacy ***
python -m spacy download en_core_web_sm

*** Install PyTorch based on your system and package manager ***
https://pytorch.org/get-started/locally/

# Setup
## Set your OPENAI_API_KEY
1. In your terminal, run `export OPENAI_API_KEY="your-api-key-here"`

## Download the Quora Question Pairs Data Set from Kaggle
1. Navigate to this URL: https://www.kaggle.com/datasets/quora/question-pairs-dataset?
2. You will need to create a Kaggle account if you do not already have one
3. Select download and load via your preference, the download zip is ~22MB
4. Unzip or simply ensure your data is in the /data folder path, this should be the questions.csv file that is about ~61MB
5. the gitignore file is set up to not push anything in the data folder to github

# Explanation of files
## main.py
Purpose: Entry point for the project. Presents a menu to select and launch different model training or evaluation modes.
How to run:
```python main.py```
Follow the prompts it provides after running.
Explanation of prompts: 

### Selection 1: Siamese Network 
This prompt will execute ```siamese_simple_main.py``` which allows you to kick off training for our Siamese Neural Network Classifiers. 

You will further be prompted to :
1. Train BERT Siamese Network
2. Train BERT+GPT Combined Network
3. Train Siamese Network with GPT Encodings

Note that these will not run unless you have the access to the authorization header of our finetuned embedding model which we do not provide for public use.
```If you are seeking to replicate our project you should first finetune your own embedding model.``` 

#### siamese_gpt_model.py
Purpose: Defines a Siamese model that uses GPT-based embeddings for question pairs.
How to run:
Imported and used by siamese_simple_main.py select option 3: Train Siamese Network with GPT Encodings

#### siamese_model_with_gpt_scores.py
Purpose: Siamese model that combines BERT embeddings with additional similarity scores (e.g., cosine similarity, Manhattan distance) that were generated from the embeddings retrieved from the Openai API
How to run:
Imported and used by siamese_simple_main.py select option 2: Train BERT+GPT Combined Network

#### siamese_model.py
Purpose: Defines the core Siamese neural network architecture and training loop using BERT-based embeddings.
How to run:
Imported and used by siamese_simple_main.py select 1: Train BERT Siamese Network

### Selection 2: GPT4 Analysis 
Runs GPT-4 based analysis on the validation set, generating predictions or embeddings using the OpenAI API.
How to run:
Select 2 when prompted.
The script will process the validation data in batches, save results, and print progress.
Note: Requires a valid OpenAI API key in your environment

#### gpt4.py
Purpose: Contains functions for interacting with the OpenAI GPT-4 API (e.g., generating embeddings or analyses).
How to run:
Imported and used by other scripts. Requires an OpenAI API key set in the environment variable OPENAI_API_KEY.

### Selection 3: Run miscDataReview.py 
Executes the ```miscDataReview.py``` script for exploratory data analysis or review. 
miscDataReview.py was a file created for post hoc investigation and is not particularly 'production ready.' We were simply modifying it as we wanted to explore specific files, compare false positives across models, execute some file clean up etc. Feel free to review this file but it is not intended for out of the box use as it relies on many files that resulted from our SNN training efforts. 

### Selection 4: 

### Selection 5: 

## postprocess_final_model.py
Purpose: Post-processes model predictions, such as applying confidence thresholds and saving adjusted results.
We generated this file to fine tune thresholds on a validation set 

How to run:
Run directly

## helper_funcs.py
Purpose: Utility functions for evaluation, metrics calculation, and other helper operations.
How to run:
Imported and used by other scripts.

Note

## preprocessData.py
Purpose: Contains functions for preprocessing the dataset, such as cleaning, normalizing, and feature engineering.
How to run:
Imported and called by other scripts (e.g., main.py or siamese_simple_main.py)

