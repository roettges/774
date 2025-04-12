import pandas as pd
import re
import numpy as np
from num2words import num2words

def preprocessing(df):
    """
    Preprocess the DataFrame by dropping unnecessary rows. 
    Removes rows with questions that are less than 10 characters or NaN
    Removes rows with questions longer than 100 words
    lowercases all questions
    modifies questions with word/word to word or word
    
    Args:
        df (pd.DataFrame): The input DataFrame to preprocess.
    
    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """

    #drop rows with NaN questions i.e. where columns question1 or question2 are NaN
    df = df.dropna(subset=['question1', 'question2'])
    
    #drop rows with questions that are less than 10 characters
    df = df[df['question1'].str.len() >= 10]
    df = df[df['question2'].str.len() >= 10]
    
    #drop rows with questions longer than 100 words
    df = df[df['question1'].str.split().str.len() <= 100]
    df = df[df['question2'].str.split().str.len() <= 100]
    
    #lowercase all questions
    df['question1'] = df['question1'].str.lower()
    df['question2'] = df['question2'].str.lower()
    
    #modify questions with word/word 
    # cases:
    # 1. I do not want to modify the slash in [/math]
    # 2. w/ w/o I want to modify to with without
    # 3. 80s/90s or his/her most cases I will simply add a space in between the characters and the slash if it is already word / word I will not modify
    # use the clean_slashes function to do this
    df['question1'] = df['question1'].apply(clean_slashes)
    df['question2'] = df['question2'].apply(clean_slashes)  
    
    #save data to csv in /data folder
    df.to_csv('data/preprocessedquestions.csv', index=False)
    return df


def clean_slashes(text):
    # first split the text by whitespace
    # check if any scenario of w/o, w/ is present
    # if so, replace it with without, with
    words = text.split()
    words = [word.replace('w/', 'with') if word.lower() == 'w/' else word for word in words]
    words = [word.replace('w/o', 'without') if word.lower() == 'w/o' else word for word in words]
    text = ' '.join(words)
    
    # Step 1: Temporarily remove [math]...[/math] blocks
    math_blocks = re.findall(r'\[math\].*?\[/math\]', text)
    placeholders = [f"__MATH_BLOCK_{i}__" for i in range(len(math_blocks))]
    
    for block, placeholder in zip(math_blocks, placeholders):
        text = text.replace(block, placeholder)

    def add_space(match):
        return f"{match.group(1)} / {match.group(2)}"
    
    text = re.sub(r'(\w)/(\w)', add_space, text)
    # Handle various word/word patterns
    text = re.sub(r'(\w+|\"\w+\")\s*/\s*(\w+|\"\w+\")', add_space, text)

    # Step 3: Put the math blocks back
    for placeholder, block in zip(placeholders, math_blocks):
        text = text.replace(placeholder, block)

    return text

def convert_numbers_outside_math(text):
    # Find math blocks and replace them with placeholders
    math_blocks = re.findall(r'\[math\].*?\[/math\]', text)
    placeholder = '@MATH@'
    temp_text = re.sub(r'\[math\].*?\[/math\]', placeholder, text)

    # Convert numbers outside math
    temp_text = re.sub(r'\b\d+\b', lambda m: num2words(int(m.group())), temp_text)

    # Put math blocks back
    for block in math_blocks:
        temp_text = temp_text.replace(placeholder, block, 1)

    return temp_text

# # Example usage
# data = pd.read_csv('data/questions.csv')
# print("Number of rows in the original data: ", len(data))
# data = preprocessing(data)
# #print how many rows are in the data
# print("Number of rows in the data: ", len(data))
