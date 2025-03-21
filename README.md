# 774
Repository for our 774 project for question matching

# Install requirements
pip install py_stringmatching numpy pandas scikit-learn

***DO NOT DO THIS JUST YET *** 
TBD but may need to also pip install nltk and then after installation, for NLTK download required data:
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

## 1st download the Quora Question Pairs Data Set from Kaggle
1. Navigate to this URL: https://www.kaggle.com/datasets/quora/question-pairs-dataset?
2. You will need to create a Kaggle account if you do not already have one
3. Select download and load via your preference, the download zip is ~22MB
4. Unzip or simply ensure your data is in the /data folder path, this should be the questions.csv file that is about ~61MB
5. the gitignore file is set up to not push anything in the data folder to github

