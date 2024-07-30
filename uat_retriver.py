import vertexai
from vertexai.preview.language_models import TextEmbeddingModel,TextEmbeddingInput
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from pinecone import Pinecone

#============================== Initialization Vertexai-TextEmbedding Model ==============================#

key_path = r"/kaggle/input/keyfile/stellarforge-0f2555ce6b2f.json"
credentials = Credentials.from_service_account_file(
    key_path,
    scopes=['https://www.googleapis.com/auth/cloud-platform'])

if credentials.expired:
    credentials.refresh(Request())
    
PROJECT_ID = 'stellarforge'
REGION = 'us-central1'
vertexai.init(project = PROJECT_ID, location = REGION, credentials = credentials)
model = TextEmbeddingModel.from_pretrained("text-embedding-004")
dimensionality = 768
kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}

#============================== Initialization Pinecone Index ==============================#

index_name = "uat-index-with-exp-decay"
pc = Pinecone(api_key="682ee7aa-7a50-48d7-beb4-c341b29fb086")
index = pc.Index(index_name)
print(f"Index {index_name} has been initialized ")

#============================== Function to get UAT branch and keywords from the query ==============================#

def uat_retriever(query,top_k=5):
    '''
    Description:   
        Function to get UAT branch and keywords from the query
    Args:
        query : str : Query text
        top_k : int : Number of results to return
    Returns:
        uat_branch_list : list : List of UAT branches
        uat_keywords_list : list : List of UAT keywords
        uat_score_dict : dict : Dictionary of UAT keywords and scores
    '''
    uat_branch_list = []
    uat_keywords_list = []
    uat_score_dict = {}
    text_embedding_input = TextEmbeddingInput(
        task_type='RETRIEVAL_QUERY', text=query
    )
    embeddings = model.get_embeddings([text_embedding_input], **kwargs)
    for embedding in embeddings:
        vector = embedding.values
    results = index.query(top_k=top_k, vector=vector,include_metadata=True, include_values=False)
    for i in results.matches:
        uat_branch_list.append(i["metadata"]['branch'])
        uat_keywords_list.append(i["metadata"]['keyword'])
        uat_score_dict[i["metadata"]['keyword']] = i["score"]
    return uat_branch_list,uat_keywords_list,uat_score_dict

