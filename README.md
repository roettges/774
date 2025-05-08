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
```If you are seeking to replicate our project you should first finetune your own embedding model.```  See out ```finetune_sbert.py``` file for an example of how we did this. 

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
Runs GPT-4 based analysis on the validation set, generating predictions based on a specific prompt asking about the question pairs' semantic similarity.  
How to run:
1. Select option 2 when prompted in main.py
2. The script will process each question pair through GPT-4o
3. For each pair, GPT outputs a Y/N decision and confidence score (0-100)
4. Results are saved with columns 'gpt_is_duplicate' and 'gpt_confidence_score'
5. Processing happens with rate limiting to respect OpenAI's API limits

If you are looking for where we did our OpenAI embeddings that is outside the main.py function, see instead the {How to Generate Embeddings} section. 

Relevant files: 
#### gpt4.py
Purpose: Contains functions for interacting with the OpenAI GPT-4 API, including:
- `openai_api_call`: Makes individual API calls to GPT-4o with formatted prompts
- `process_api_response`: Parses GPT responses into binary decisions and confidence scores
- `gpt4_analysis`: Batch processes dataframes through GPT with rate limiting

Implementation details:
- Uses system message to enforce output format: "[Y|N] [confidence score]"
- Handles errors and malformed responses gracefully
- Implements rate limiting (300 requests/minute)
- Logs progress every 200 rows

Requirements:
- Valid OpenAI API key set as environment variable
- openai Python package installed
- Internet connection for API access

### Selection 3: Run miscDataReview.py 
Executes the ```miscDataReview.py``` script for exploratory data analysis or review. 
miscDataReview.py was a file created for post hoc investigation and is not particularly 'production ready.' We were simply modifying it as we wanted to explore specific files, compare false positives across models, execute some file clean up etc. Feel free to review this file but it is not intended for out of the box use as it relies on many files that resulted from our SNN training efforts. 

### Selection 4: Save Similarity Scores
This mode lets you calculate and save various similarity metrics for question pairs. We started with this as we were still sorting out some of our direction for our work, however this section utilizes helper functions that explored stop word removal which was relevant when we were determining if stop word removal would be a worthwhile preprocessing step or not. 

You do not really need this option in order to conduct our overall experiences but we are including them for reference. 

#### How to Run:
1. Select option 4 when prompted in main.py
2. Choose from the following sub-selections:
   - 1: Jaccard Similarity
   - 2: TF-IDF and Cosine Similarity (DO NOT USE THIS WAS NOT IMPLEMENTED AS IT WAS OUT OF SCOPE)
   - 3: Levenshtein Distance

#### Jaccard Similarity Options:
When choosing Jaccard Similarity, you'll be prompted to:
1. Select a parsing method:
   - 1: 3-gram tokenization (character level)
   - 2: Whitespace tokenization (word level)
   - 3: Lemmatized word tokenization
2. Choose whether to remove stopwords

#### Levenshtein Distance Options:
When choosing Levenshtein Distance, you'll be prompted to:
1. Choose whether to remove stopwords

#### Implementation Details:
The implementation uses `helper_funcs.py` which provides:
- `saveJaccard()`: Calculates Jaccard similarity using different tokenization strategies
This is really where we explored stop word removal. 
- `saveLevenshtein()`: Calculates edit distance between questions
- `clean_data()`: Preprocesses text by lowercasing and optionally removing stopwords
- `saveData()`: Saves results in both CSV and compressed pickle formats
- `timer` decorator: Measures and reports function execution time

Results are saved to the `output` directory as both CSV and pickle files with the appropriate naming convention based on the chosen method and parameters.

### Selection 5: Distance Metrics for GPT4 Embeddings
This mode calculates various distance metrics between question pair embeddings generated from OpenAI's embedding model.

#### How to Run:
1. Select option 5 when prompted in main.py
2. The script will look for embeddings in the expected location: "data/embeddings_and_data_with_embeddings/output_with_embeddings.csv"
3. For each question pair, it calculates:
   - Cosine similarity 
   - Manhattan distance
   - Euclidean distance
4. Results are saved to two files:
   - CSV: "data/embeddings_and_data_with_embeddings/full_data_with_gpt4embeddings_and_distances.csv"
   - Pickle: "data/embeddings_and_data_with_embeddings/full_data_with_gpt4embeddings_and_distances.pkl"

#### Implementation Details:
The script processes embeddings row by row, extracting vectors from string representation and applying different similarity/distance measures from helper_funcs.py:
- `cosine_similarity()`: Measures the cosine of the angle between vectors (1 = identical, 0 = orthogonal)
- `manhattan_distance()`: Calculates L1 norm (sum of absolute differences)
- `euclidean_distance()`: Calculates L2 norm (straight-line distance)

#### Note:
This process requires pre-generated embeddings from the OpenAI API. See the "OpenAI Embeddings Generation" section for details on creating these embeddings.

### Selection 6: Model Evaluation
This mode provides comprehensive evaluation metrics for model predictions, allowing for threshold tuning and detailed error analysis.

#### How to Run:
1. Select option 6 when prompted in main.py
2. Choose between two evaluation paths:
   - Option 1: Evaluate GPT4 embeddings with cosine similarity
   - Option 2: Evaluate any model output file

#### GPT4 Embeddings Evaluation:
1. Loads embeddings with distances from "data/embeddings_and_data_with_embeddings/full_data_with_gpt4embeddings_and_distances.csv"
2. Prompts for a cosine similarity threshold
3. Uses this threshold to convert similarity scores to binary predictions
4. Evaluates against ground truth and generates metrics

#### Custom Model Evaluation:
1. Prompts for:
   - Filename of model predictions
   - Column name for ground truth (defaults to "is_duplicate")
   - Column name for predictions or similarity scores
   - Threshold to apply (if using similarity scores)
2. Runs full evaluation using helper_funcs.evaluateM()

#### Evaluation Outputs:
For both evaluation paths, the script:
1. Calculates metrics: accuracy, precision, recall, F1-score
2. Saves metrics to "evaluation_results/[filename]/evaluation_results.txt"
3. Generates confusion matrix and ROC curve visualizations
4. Identifies and saves false positives and false negatives for error analysis

#### Implementation:
Uses helper_funcs.py which provides:
- `simpleEvaluateM()`: Basic metrics without visualizations
- `evaluateM()`: Comprehensive evaluation with visualizations

## postprocess_final_model.py
Purpose: Post-processes model predictions by applying confidence thresholds to raw prediction scores.

### Functionality:
- Allows fine-tuning decision thresholds on validation/test sets without retraining models
- Reclassifies predictions based on confidence thresholds
- Generates updated evaluation metrics
- Particularly useful for optimizing precision/recall tradeoffs

### Key Functions:
- `postprocess_final_model(file, thresh)`: Takes prediction file and confidence threshold, applies threshold to raw prediction scores
- Adjusts prediction classification: 
  - Predictions with confidence below threshold → classified as non-duplicates (0)
  - Predictions with confidence above threshold → retain original classification

How to run:
Run directly

## helper_funcs.py
Purpose: Utility functions for evaluation, metrics calculation, and other helper operations.
How to run:
Imported and used by other scripts.

See more specifics noted in the explanation of main.py

## preprocessData.py
Purpose: Handles data preprocessing for question pairs, including cleaning, normalization, and specialized text transformations.

### Main Functions:
- `preprocessing(df)`: Primary preprocessing pipeline that:
  - Removes rows with NaN values in question fields
  - Filters out questions less than 10 characters
  - Removes questions with more than 100 words
  - Lowercases all questions
  - Handles special slash formats (w/, w/o, word/word)
  - Saves preprocessed data to 'data/preprocessedquestions.csv'

- `clean_slashes(text)`: Specialized function that:
  - Preserves math notation (e.g., [math]...[/math] blocks)
  - Converts "w/" to "with" and "w/o" to "without"
  - Adds spaces around slashes for word/word constructs
  - Handles quoted words and complex patterns
  
- `convert_numbers_outside_math(text)`: Converts numeric digits to words:
  - Preserves numbers inside math notation
  - Converts all other numbers to word form using num2words

### How to Run:
Imported and called by other scripts (e.g., main.py or siamese_simple_main.py)
Can be run directly to preprocess the full dataset:
```bash
python preprocessData.py
```

## OpenAI Embeddings Generation
Our project uses OpenAI's API to generate embeddings for question pairs. The embeddings workflow is implemented in the `embeddings` directory.
- ```embedding_main.py```: Main script orchestrating the embedding workflow
- ```create_batch_files.py```: Creates JSONL files for batch processing
- ```embedding_api_call.py```: Handles API communication with OpenAI
- ```check_batch_runs.py```: Monitors batch processing status

### How to Generate Embeddings
The workflow creates 512-dimensional embeddings using the text-embedding-3-large model. ```

1. **Prepare Environment**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
2. Set up batch processing 
- Open ```embedding_main.py``` uncomment the batch creation and submission code 
- set data_file_path to your questions CSV 
- adjust batch_size if needed (default: 40000 requests)
3. Create and Submit Batches 
    ```bash
    cd embeddings
    mkdir -p batch_files
    python embedding_main.py
    ```
4. Check batch status
    ```bash 
    python check_batch_runs.py
    ```
5. Process Results Once batches are complete:
    -Update the completed_batches dictionary in embedding_main.py with batch-to-file mappings
    Run python embedding_main.py again to:
        - Download all embeddings results
        - Add embeddings to your dataset
        - Save as output_with_embeddings.csv
