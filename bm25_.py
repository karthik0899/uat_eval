import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import huggingface_hub
from vertexai_llm import get_response

huggingface_hub.login(token='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')
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-Nemo-Instruct-2407")

df_uat = pd.read_csv("UAT.csv")
df_uat = df_uat.replace(np.nan, '', regex=True)
df_embeddings = df_uat.copy(deep=True)
unique_values = list(set(df_uat.values.flatten()))
unique_values.remove('')

list_of_lists = df_uat.apply(lambda row: [value for value in row if value != ''], axis=1).tolist()


list_of_strings = [' '.join([str(elem) for elem in sublist]) for sublist in list_of_lists]


from rank_bm25 import BM25Okapi
import numpy as np

corpus = list_of_strings

def position_weighted_tokens(text):
    tokens = tokenizer.tokenize(text)
    weights = [1 + (i / len(tokens)) for i in range(len(tokens))]
    return [(token, weight) for token, weight in zip(tokens, weights)]

class WeightedBM25Okapi(BM25Okapi):
    def __init__(self, corpus):
        weighted_corpus = [position_weighted_tokens(text) for text in corpus]
        super().__init__([
            [token for token, _ in doc] for doc in weighted_corpus
        ])
        self.doc_weights = [
            {token: weight for token, weight in doc} for doc in weighted_corpus
        ]

    def get_scores(self, query):
        scores = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([doc.get(q, 0) for doc in self.doc_freqs])
            q_weights = np.array([doc.get(q, 0.001) for doc in self.doc_weights])
            scores += (self.idf.get(q, 0) * q_freq * q_weights * (self.k1 + 1)
                       / (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return scores

# Create weighted BM25 object
weighted_bm25 = WeightedBM25Okapi(corpus)

def search(query, n=5):
    query_tokens = tokenizer.tokenize(query)
    scores = weighted_bm25.get_scores(query_tokens)
    
    # Get top N indices
    top_n_indices = np.argsort(scores)[-n:][::-1]
    
    # Return top N retrievals with their scores
    top_n_retrievals = [
        {"text": corpus[i], "score": scores[i]}
        for i in top_n_indices
    ]
    
    return top_n_retrievals


output = get_response(prompt_name="keyword_extract", input_data = "abstract")

import json

def clean_json_string(input_string):
    lines = input_string.split('\n')
    start = 0
    end = len(lines)

    # Remove opening ```json if present
    if lines[0].strip() == '```json':
        start = 1

    # Remove closing ``` if present
    if lines[-1].strip() == '```':
        end = -1

    return '\n'.join(lines[start:end])
# Usage:
json_string = clean_json_string(output)
concepts_ = json.loads(json_string)

list_of_terms = [concept['term'] for concept in concepts_['concepts']]
# list_of_terms = tokenizer.tokenize('Measuring the relation between star formation and galactic winds is observationally difficult. In this work we make an indirect measurement of the mass-loading factor (the ratio between the mass outflow rate and star formation rate) in low-mass galaxies using a differential approach to modeling the low-redshift evolution of the star-forming main sequence and mass–metallicity relation. We use Satellites Around Galactic Analogs (SAGA) background galaxies, i.e., spectra observed by the SAGA Survey that are not associated with the main SAGA host galaxies, to construct a sample of 11,925 spectroscopically confirmed low-mass galaxies from 0.01 ≲ z ≤ 0.21 and measure auroral line metallicities for 120 galaxies. The crux of the method is to use the lowest-redshift galaxies as the boundary condition of our model, and to infer a mass-loading factor for the sample by comparing the expected evolution of the low-redshift reference sample in stellar mass, gas-phase metallicity, and star formation rate against the observed properties of the sample at higher redshift. We infer a mass-loading factor of ${\eta }_{{\rm{m}}}={0.92}_{-0.74}^{+1.76}$ , which is in line with direct measurements of the mass-loading factor from the literature despite the drastically different sets of assumptions needed for each approach. While our estimate of the mass-loading factor is in good agreement with recent galaxy simulations that focus on resolving the dynamics of the interstellar medium, it is smaller by over an order of magnitude than the mass-loading factor produced by many contemporary cosmological simulations.')
terms_string = ' '.join(list_of_terms)
print(terms_string)
# Example usage
top_results = search(terms_string, n=20)




