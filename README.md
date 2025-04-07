# 774
Repository for our 774 project for question matching

# Install requirements
pip install py_stringmatching numpy pandas scikit-learn openai
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


***DO NOT DO THIS JUST YET *** 
TBD but may need to also pip install nltk and then after installation, for NLTK download required data:
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Setup
## Set your OPENAI_API_KEY
1. In your terminal, run `export OPENAI_API_KEY="your-api-key-here"`
## Download the Quora Question Pairs Data Set from Kaggle
1. Navigate to this URL: https://www.kaggle.com/datasets/quora/question-pairs-dataset?
2. You will need to create a Kaggle account if you do not already have one
3. Select download and load via your preference, the download zip is ~22MB
4. Unzip or simply ensure your data is in the /data folder path, this should be the questions.csv file that is about ~61MB
5. the gitignore file is set up to not push anything in the data folder to github
