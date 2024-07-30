# Similarity Search

This module provides functionality for performing similarity searches on a dataset of scientific abstracts and titles, as well as forecasting trends based on the search results.

## Installation

To use this module, you'll need to have Python 3.x installed, along with the following dependencies:

- numpy
- pandas
- faiss
- torch
- sentence_transformers
- pmdarima
- huggingface_hub

You can install these dependencies using pip:


pip install numpy pandas faiss torch sentence_transformers pmdarima huggingface_hub


## Usage

### Similarity Search

The `search` function performs a similarity search on the dataset based on a given query string. It returns the most similar abstract and title.


from similarity_search import search

abstract, title = search("query string", k=5)


- `query` (str): The query string to search for.
- `k` (int, optional): The number of top results to return. Defaults to 5.

### Trend Search

The `trend_search` function performs a similarity search and returns the top-k most similar datapoints with trends for a specified forecasting type (read_count, citation_count, or cite_read_boost).


from similarity_search import trend_search

results = trend_search("query string", "read_count", k=5)


- `query` (str): The query string to search for.
- `forecasting_type` (str): The type of data to forecast, either "read_count", "citation_count", or "cite_read_boost".
- `k` (int, optional): The number of top results to return. Defaults to 5.

### Forecasting

The `ADSTrends_forecast` function ties together the trend search, preprocessing, forecasting, and visualization based on the input parameters.


from similarity_search import ADSTrends_forecast

date_list, preds, uci, lci, conf = ADSTrends_forecast("query string", "read_count", max_samples=1000, n_months=3)


- `text` (str): The query text for searching the dataset.
- `forecasting_type` (str): The type of data to forecast, either "read_count", "citation_count", or "cite_read_boost".
- `max_samples` (int, optional): The maximum number of samples to consider from the trend_search results. Defaults to 1000.
- `n_months` (int, optional): The number of future months to forecast. Defaults to 3.

### Keyword Search

The `keyword_search` function performs a keyword search on the dataset and returns the top-k most similar abstracts, titles, keywords, and similarity scores.


from similarity_search import keyword_search

results = keyword_search("query string", k=5)


- `query` (str): The query string to search for.
- `k` (int, optional): The number of top results to return. Defaults to 5.

### Keyword Generation

The `ADS_KeywordGen` function generates the top-n most frequent keywords from the given text.


from similarity_search import ADS_KeywordGen

top_keywords = ADS_KeywordGen("text", top_n=5)


- `text` (str): The input text.
- `top_n` (int, optional): The number of top keywords to return. Defaults to 5.

## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

