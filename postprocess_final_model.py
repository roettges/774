import pandas as pd
import re
import numpy as np
import helper_funcs as hf

#file = 'predictions_2025-04-18_15-34-56.csv'
file = 'FINAL_TEST_predictions_2025-05-02_16-55-12.csv'
threshold = 0.88465

def postprocess_final_model(fp, thresh):
    '''Given the file containing the final model, re-predict the predictions favoring high confidence so that we take a more cautious approach to matches'''
    df = pd.read_csv(fp)
    # Check if the required columns are present
    required_columns = ['raw_prediction', 'predicted']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input file must contain columns: {required_columns}")
    # Initialize a new column for adjusted predictions
    df['adjusted_prediction'] = df['predicted']
    # Iterate through the DataFrame and adjust predictions based on confidence threshold
    for index, row in df.iterrows():
        raw_prediction = row['raw_prediction']
        predicted = row['predicted']
        
        # If the raw prediction is above the threshold, keep the original prediction
        if abs(raw_prediction) >= thresh:
            df.at[index, 'adjusted_prediction'] = predicted
        # If the raw prediction is below the threshold, set the adjusted prediction to 0
        else:
            df.at[index, 'adjusted_prediction'] = 0
    return df

# Call the function and get the adjusted predictions
adjusted_df = postprocess_final_model(file, threshold)
# Save the adjusted predictions to a new CSV file
#output_file = f'postprocessed_predictions_{threshold}.csv'
output_file = f'{file.split(".")[0]}_postprocessed_predictions_{threshold}.csv'
adjusted_df.to_csv(output_file, index=False)
hf.simpleEvaluateM(output_file, 'is_duplicate', 'adjusted_prediction')
    
    