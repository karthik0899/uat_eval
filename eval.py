from uat_suggestor import uat_manager
from google.cloud import storage
import pandas as pd
import numpy as np
import os

df = pd.read_csv('uat_eval_dataset.csv')

df_subset = df.sample(n=1000)
df_subset = df_subset.reset_index(drop=True)


import time

# List of columns we expect to have or create
expected_columns = [
    'type', 
    'output', 
    'list_of_branch', 
    'retreived_branches_abstract', 
    'retreived_branches_concepts', 
    'retreived_keywords_abstract'
]

# Create a copy of the original dataframe
df_subset_eval = df_subset.copy()

# Create missing columns
for column in expected_columns:
    if column not in df_subset_eval.columns:
        df_subset_eval[column] = None

# Process the dataframe
for i in range(len(df_subset_eval['abstract'])):
    max_attempts = 3
    attempt = 0
    success = False

    while attempt < max_attempts and not success:
        try:
            response = uat_manager(df_subset_eval['abstract'][i])            
            df_subset_eval.at[i, 'type'] = response['type']
            df_subset_eval.at[i, 'output'] = response['output']
            print(len(response['list_of_branch']))
            df_subset_eval.at[i, 'list_of_branch'] = response['list_of_branch']
            df_subset_eval.at[i, 'retreived_branches_abstract'] = response['retreived_branches_abstract']
            df_subset_eval.at[i, 'retreived_branches_concepts'] = response['retreived_branches_concepts']
            df_subset_eval.at[i, 'retreived_keywords_abstract'] = response['retreived_keywords_abstract']

            success = True
        except Exception as e:
            attempt += 1
            if attempt < max_attempts:
                print(f"Error occurred: {str(e)}. Retrying... (Attempt {attempt} of {max_attempts})")
                time.sleep(1)  # Wait for 1 second before retrying
            else:
                print(f"Error persisted after {max_attempts} attempts: {str(e)}")
                error_message = f"Error after {max_attempts} attempts: {str(e)}"
                for column in expected_columns:
                    df_subset_eval.at[i, column] = error_message

    # Optional: Print progress
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1} out of {len(df_subset_eval)} rows")
        
        


df_subset_eval.to_csv('uat_eval_dataset_output_new.csv', index=False)


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'stellarforge-0f2555ce6b2f.json'
os.environ['PROJECT_ID'] = 'stellarforge'

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )

# Usage
bucket_name = "eval_dataset_sf"
source_file_name = "uat_eval_dataset_output_new.csv"
destination_blob_name = "uat_eval_dataset_output.csv"

upload_blob(bucket_name, source_file_name, destination_blob_name)