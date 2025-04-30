import pandas as pd
# get the length of data/questions.csv excluding the header line
df = pd.read_csv('data/questions.csv')
print(f"Number of rows in questions.csv: {len(df)}")  
#get count of rows where is_duplicate is 1 
df_duplicates = df[df['is_duplicate'] == 1]
df_nondups = df[df['is_duplicate'] == 0]
print(f"Number of rows in questions.csv where is_duplicate is 1: {len(df_duplicates)}")
print(f"Number of rows in questions.csv where is_duplicate is 0: {len(df_nondups)}")

df_preprocessed = pd.read_csv('data/preprocessedquestions.csv')
print(f"Number of rows in preprocessedquestions.csv: {len(df_preprocessed)}")  
#get count of rows where is_duplicate is 1
pre_df_duplicates = df_preprocessed[df_preprocessed['is_duplicate'] == 1]
pre_df_nondups = df_preprocessed[df_preprocessed['is_duplicate'] == 0]
print(f"Number of rows in preprocessedquestions.csv where is_duplicate is 1: {len(pre_df_duplicates)}")
print(f"Number of rows in preprocessedquestions.csv where is_duplicate is 0: {len(pre_df_nondups)}")

# Load and preprocess DataFrames
df_jaccard_3gram = pd.read_csv('data/questions.csv')
df_jaccard_lemma = pd.read_csv('output/jaccard_lemma.csv')

# Convert questions to lowercase
for df in [df_jaccard_3gram, df_jaccard_lemma]:
    df['question1'] = df['question1'].str.lower()
    df['question2'] = df['question2'].str.lower()

# Merge on id and compare questions
merged_df = df_jaccard_3gram.merge(
    df_jaccard_lemma, 
    on='id', 
    suffixes=('_3gram', '_lemma')
)

# Find rows where questions differ (case-insensitive)
different_rows = merged_df[
    (merged_df['question1_3gram'].str.lower() != merged_df['question1_lemma'].str.lower()) |
    (merged_df['question2_3gram'].str.lower() != merged_df['question2_lemma'].str.lower())
]

print(f"Number of rows with different questions (ignoring case): {len(different_rows)}")

# Print first 50 differences
for idx, row in different_rows.head(50).iterrows():
    print(f"\nID: {row['id']}")
    if row['question1_3gram'].lower() != row['question1_lemma'].lower():
        print("Question 1 differs:")
        print(f"3gram: {row['question1_3gram']}")
        print(f"Lemma: {row['question1_lemma']}")
    if row['question2_3gram'].lower() != row['question2_lemma'].lower():
        print("Question 2 differs:")
        print(f"3gram: {row['question2_3gram']}")
        print(f"Lemma: {row['question2_lemma']}")
        
# identify row ids that existed in the non finetuned bert false positive files but don't exist in the finetuned bert false positive files
# create a new subfolder under evaluation_results called FP_comparison and make a file indicating 


# identify 'id' values in evaluation_results/siamese_finetuned_onlinecontrastive_preprocessed_predictions_2025-04-15_11-08-11/false_positive/false_positive_examples.csv that are not in evaluation_results/gpt_cos_threshold/0.83/false_positive/false_positive_examples.csv
# then do the inverse
# create a new subfolder under evaluation_results called FP_comparison and make a file indicating the differences
#gpt_fp = pd.read_csv('evaluation_results/gpt_cos_threshold/0.83/false_positive/false_positive_examples.csv')
gpt_fp = pd.read_csv('evaluation_results/siamese_gpt_predictions_REMOVED_CORRUPTIONS/false_positive/false_positive_examples.csv')
siamese_fp = pd.read_csv('evaluation_results/siamese_finetuned_onlinecontrastive_preprocessed_predictions_2025-04-15_11-08-11/false_positive/false_positive_examples.csv')
gpt_fp_ids = set(gpt_fp['id'])
siamese_fp_ids = set(siamese_fp['id'])
# find ids in gpt_fp that are not in siamese_fp
gpt_fp_not_in_siamese = gpt_fp[~gpt_fp['id'].isin(siamese_fp_ids)]
# find ids in siamese_fp that are not in gpt_fp
siamese_fp_not_in_gpt = siamese_fp[~siamese_fp['id'].isin(gpt_fp_ids)]
# create a new directory called FP_comparison
import os
os.makedirs('evaluation_results/FP_comparison', exist_ok=True)
# save the results to a csv file not just the id column but also the question1 and question2 columns
gpt_fp_not_in_siamese[['id', 'question1', 'question2']].to_csv('evaluation_results/FP_comparison/gpt_fp_not_in_siamese.csv', index=False)
siamese_fp_not_in_gpt[['id', 'question1', 'question2']].to_csv('evaluation_results/FP_comparison/siamese_fp_not_in_gpt.csv', index=False)
# get counts of overlapping ids, and unique ids for each
gpt_fp_not_in_siamese_count = len(gpt_fp_not_in_siamese)
siamese_fp_not_in_gpt_count = len(siamese_fp_not_in_gpt)
print(f"Number of false positives in gpt_fp not in siamese_fp: {gpt_fp_not_in_siamese_count}")
print(f"Number of false positives in siamese_fp not in gpt_fp: {siamese_fp_not_in_gpt_count}")
# get the total number of false positives in each
# gpt_fp_count = len(gpt_fp)
# siamese_fp_count = len(siamese_fp)
# print(f"Total number of false positives in gpt_fp: {gpt_fp_count}")
# print(f"Total number of false positives in siamese_fp: {siamese_fp_count}")
# get the total number of unique ids across both files
gpt_fp_ids = set(gpt_fp['id'])
siamese_fp_ids = set(siamese_fp['id'])
total_unique_ids = len(gpt_fp_ids.union(siamese_fp_ids))
print(f"Total number of unique ids across both files: {total_unique_ids}")
# number in both
gpt_fp_and_siamese_count = len(gpt_fp_ids.intersection(siamese_fp_ids))
print(f"Number of false positives in both files: {gpt_fp_and_siamese_count}")


