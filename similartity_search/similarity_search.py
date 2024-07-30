import os
import ast
import faiss
import torch
import numpy as np
import pandas as pd
from datetime import date
from typing import List , Tuple
from collections import Counter
from pmdarima.arima import auto_arima
from sentence_transformers import SentenceTransformer
from huggingface_hub import login, hf_hub_download

# Placeholder for an environment token
token = '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'
# Setup for token usage and model loading
if token is not None:
    login(token=token)
else:
    print("HF 🤗 Token not found in environment variables!")


# Define the repository for downloading embeddings
REPO_ID = "Swarnava/NASA_ADS"

# Ensure the embedding path directory exists
os.makedirs('Persistance_storage', exist_ok=True)

# Filenames to download
files_to_download = [("embeddings/embeddings.npy",'Persistance_storage'), ("embedding_data.csv","Persistance_storage/embedding_data")]

# Function to check and download each file
def download_file(file_details):
    local_file_path = os.path.join(file_details[1], file_details[0])
    if not os.path.exists(local_file_path):
        try:
            hf_hub_download(repo_id=REPO_ID, filename=file_name[0], repo_type="dataset", local_dir=file_details[1],local_dir_use_symlinks=False)
            print(f"Downloaded {file_details[0]} successfully.")
        except Exception as e:
            print(f"Error occurred during the download of {file_details[0]}: {e}")
            print("You might not have access to the Dataset or the file might not exist.")
    else:
        print(f"{file_details[0]} already exists, no need to download.")

# Download each file if not already present
for file_name in files_to_download:
    download_file(file_name)

# Load and preprocess data efficiently
embeddings = np.load(os.path.join('Persistance_storage', "embeddings/embeddings.npy"))
df = pd.read_csv(os.path.join('Persistance_storage', "embedding_data/embedding_data.csv"))
df['read_count'] = df['read_count'].fillna(0)
df['cite_read_boost'] = df['cite_read_boost'].fillna(0)
df = df[(df.title_length > 5) & (df.title_length < 20) & (df.abstract_length > 50) & (df.abstract_length < 400)]
df = df[['title', 'abstract', 'keyword', 'year', 'date', 'read_count', 'cite_read_boost']]
df.reset_index(inplace=True, drop=False)
df.rename(columns={'index': 'id'}, inplace=True)

# Load sentence transformer model
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens', device='cpu', cache_folder='Persistance_storage/cache/')

# Setup FAISS index
df_to_index = df.set_index(['id'], drop=False)
id_index = np.array(df_to_index.id.values).flatten().astype('int')
del df
normalized_embeddings = embeddings.copy()
faiss.normalize_L2(normalized_embeddings)
index_flat = faiss.IndexFlatIP(len(embeddings[0]))
del embeddings  # Free up memory
index_content = faiss.IndexIDMap(index_flat)
index_content.add_with_ids(normalized_embeddings, id_index)
del normalized_embeddings  # Free up memory after use

def search(query: str, k: int = 5,usemodel=model) -> Tuple[str, str]:
    vector = usemodel.encode([query])
    faiss.normalize_L2(vector)

    top_k = index_content.search(vector, k)
    ids = top_k[1][0].tolist()
    similarities = top_k[0][0].tolist()


    results = df_to_index.loc[ids]
    results['similarity'] = similarities
    output = results.reset_index(drop = True)[['id', 'abstract','title', 'similarity']]
    newabstract = output['abstract'][0]
    newtitle = output['title'][0]


    return (newabstract, newtitle)



def trend_search(query: str, forecasting_type: str, k: int = 5) -> pd.DataFrame:
    """
    Performs a search on the dataset and returns the top-k most similar datapoints with trends.

    Args:
        query (str): The query string.
        forecasting_type (str): The type of data to forecast, either "read_count", "citation_count", or "cite_read_boost".
        k (int, optional): The number of top results to return. Defaults to 5.

    Returns:
        pd.DataFrame: A DataFrame containing the top-k trend results.
    """

    vector = model.encode([query])
    faiss.normalize_L2(vector)
    top_k = index_content.search(vector, k)
    ids = top_k[1][0].tolist()
    similarities = top_k[0][0].tolist()

    # print(f'Searching for "{query}"...')
    results = df_to_index.loc[ids]
    results['similarity'] = similarities

    if forecasting_type == "read_count":
        columns = ['year','date', 'read_count', 'similarity']
        results = results[columns]
        results = results.dropna(subset=['read_count'])
    elif forecasting_type == "citation_count":
        columns = ['date', 'citation_count', 'similarity']
        results = results[columns]
        results = results.dropna(subset=['citation_count'])
    elif forecasting_type == "cite_read_boost":
        columns = ['date', 'cite_read_boost', 'similarity']
        results = results[columns]
        results = results.dropna(subset=['cite_read_boost'])
    else:
        raise ValueError(f"Invalid forecasting_type: {forecasting_type}")

    return results.reset_index(drop=True)

def preprocess_data(data, col):
    """
    Preprocesses the input data for forecasting by converting the date column to datetime format,
    resampling to monthly frequency, and forward/backward filling missing values.

    Args:
        data (pd.DataFrame): The input data.
        col (str): The column to preprocess.

    Returns:
        pd.Series: The preprocessed data.
    """
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    data = data.resample('ME').ffill().bfill()
    return data[col]

def predict_next_months(data, n_months=3):
    """
    Uses the auto_arima function from the pmdarima library to predict the next n_months of data
    based on the input time series data.

    Args:
        data (pd.Series): The input time series data.
        n_months (int, optional): The number of future months to predict. Defaults to 3.

    Returns:
        Tuple[pd.DatetimeIndex, np.ndarray]: A tuple containing the future dates and corresponding predictions.
    """
    model = auto_arima(data, seasonal=False, trace=False)
    # last_date = data.index.max()
    last_date = date.today()
    start_date = last_date + pd.DateOffset(months=1)
    future_dates = pd.date_range(start=start_date, periods=n_months, freq='ME')
    forecasts, confint = model.predict(n_periods=n_months, X=future_dates, return_conf_int=True)
    return future_dates, np.ceil(forecasts), np.round(confint)
  

def ADSTrends_forecast(text, forecasting_type="read_count", max_samples=1000, n_months=3):
    """
    The main function that ties together the other functions to perform the trend_search, preprocessing,
    forecasting, and visualization based on the input parameters.

    Args:
        text (str): The query text for searching the dataset.
        forecasting_type (str): The type of data to forecast, either "read_count", "citation_count", or "cite_read_boost".
        max_samples (int, optional): The maximum number of samples to consider from the trend_search results. Defaults to 1000.
        n_months (int, optional): The number of future months to forecast. Defaults to 3.
    """
    df_ = trend_search(text, forecasting_type, max_samples)
    # df_ = df_[df_['year'] != 2024]
    conf = np.round(df_.similarity.mean()*100,2)
    df_ = df_.groupby('date')[forecasting_type].mean().reset_index()
    df_[forecasting_type] = df_[forecasting_type].apply(lambda x: np.ceil(x))
    processed_data = preprocess_data(df_, forecasting_type)
    future_dates, predictions, confint = predict_next_months(processed_data, n_months)
    date_list = list(future_dates.strftime('%Y-%m-%d'))
    preds = list(predictions.values)
    lci = list(confint[:, 0])
    uci = list(confint[:, 1])
    return date_list, preds, uci, lci, conf



def keyword_search(query: str, k: int = 5) -> pd.DataFrame:
    """
    Performs a keyword search on the dataset and returns the top-k most similar abstracts, titles, keywords, and similarity scores.

    Args:
        query (str): The query string.
        k (int, optional): The number of top results to return. Defaults to 5.

    Returns:
        pd.DataFrame: A DataFrame containing the top-k results with columns 'id', 'abstract', 'title', 'keyword', and 'similarity'.
    """
    vector = model.encode([query])
    faiss.normalize_L2(vector)
    top_k = index_content.search(vector, k)
    ids = top_k[1][0].tolist()
    similarities = top_k[0][0].tolist()
    # print(f'Searching for "{query}"...')
    results = df_to_index.loc[ids]
    results['similarity'] = similarities
    results = results[['id', 'abstract', 'title', 'keyword', 'similarity']]
    results = results.dropna(subset=['keyword'])
    return results.reset_index(drop=True)

    
def ADS_KeywordGen(text: str, top_n: int = 5) -> List[str]:
    """
    Generates the top-n most frequent keywords from the given text.

    Args:
        text (str): The input text.
        top_n (int, optional): The number of top keywords to return. Defaults to 5.

    Returns:
        List[tuple]: A list of tuples containing the top keywords and their frequencies.
    """
    df_ = keyword_search(text, 100)
    mainlist = df_.keyword.values.tolist()
    word_lists = [ast.literal_eval(lst_str) for lst_str in mainlist]
    flat_list = [word for sublist in word_lists for word in sublist]
    word_frequency = Counter(flat_list)
    top_keywords = word_frequency.most_common(top_n)
    print([*dict(top_keywords)])
    return [*dict(top_keywords)]



def search_uat(query: str, k: int = 5,usemodel=model) -> Tuple[str, str]:
    vector = usemodel.encode([query])
    faiss.normalize_L2(vector)

    top_k = index_content.search(vector, k)
    ids = top_k[1][0].tolist()
    similarities = top_k[0][0].tolist()


    results = df_to_index.loc[ids]
    results['similarity'] = similarities
    output = results.reset_index(drop = True)[['id', 'abstract','title', 'similarity']]
    newabstract = output['abstract']
    ids = output['id']
    titles = output['title']
    scores = output['similarity']


    return (newabstract,ids,titles,scores)