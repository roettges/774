import os
import time
import openai
import pandas as pd

openai.api_key = os.getenv("OPENAI_API_KEY")

def process_api_response(response):
    split_response = response.split()
    
    # pre-processing
    if len(split_response) != 2:
        return None
    if split_response[0] not in ["Y", "N"]:
        return None

    # check if the second part is a valid integer (confidence score)
    try:
        confidence_score = int(split_response[1])
        if not (0 <= confidence_score <= 100):
            return None
    except ValueError:
        return None
    
    return (1 if split_response[0] == "Y" else 0, split_response[1])

def openai_api_call(q1, q2):

    prompt_content = f"Are the following two questions semantically similar?\n1. {q1}\n2. {q2}"

    completion = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "developer",
                "content": "Respond only in the format '[Y|N] [confidence score]'. The confidence score should be a number from 0-100, inclusive."
            },
            {
                "role": "user",
                "content": prompt_content
            }
        ]
    )

    return process_api_response(completion.choices[0].message.content)

def gpt4_analysis(df):

    ################## GPT RATE LIMIT ##################
    rate_limit = 300  # requests per minute
    time_period = 60  # seconds (1 minute)
    # Calculate sleep time in seconds per request
    sleep_time = time_period / rate_limit
    ####################################################

    is_duplicate = []
    confidence_scores = []
    
    row_count = 0
    for _, row in df.iterrows():
        
        if row_count % 200 == 0:
            print(f"Currently on row {row_count} at {time.strftime('%H:%M:%S')}")
        row_count += 1
        
        q1 = row['question1']
        q2 = row['question2']
        
        try:
            result = openai_api_call(q1, q2)
            
            if result:
                duplicate, confidence = result
                is_duplicate.append(duplicate)
                confidence_scores.append(confidence)
            else:
                # if gpt response is badly formatted
                is_duplicate.append(None)
                confidence_scores.append(None)
                
        except Exception as e:
            # if api call fails
            print(f"Error on row {row_count}: {e}")
            is_duplicate.append(None)
            confidence_scores.append(None)
        
        ################## GPT RATE LIMIT ##################
        time.sleep(sleep_time)
        ####################################################
    
    df['gpt_is_duplicate'] = is_duplicate
    df['gpt_confidence_score'] = confidence_scores
    
    return df