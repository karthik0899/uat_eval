import vertexai
from vertexai.generative_models import GenerativeModel,HarmBlockThreshold,HarmCategory
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from uat_eval.common.prompt_manager.PromptManager import ModelPrompt

#============================== Initialization prompt Manager ==============================#

prompt_manager = ModelPrompt()

#============================== Initialization Vertexai-Gemini Model ==============================#
key_path = r"/kaggle/input/keyfile/stellarforge-0f2555ce6b2f.json"
credentials = Credentials.from_service_account_file(
    key_path,
    scopes=['https://www.googleapis.com/auth/cloud-platform'])

if credentials.expired:
    credentials.refresh(Request())
    
PROJECT_ID = 'stellarforge'
REGION = 'us-central1'
vertexai.init(project = PROJECT_ID, location = REGION, credentials = credentials)
model = GenerativeModel(model_name="gemini-1.5-flash-001",)

#============================== Function to get response from Vertexai-Gemini Model ==============================#
 
def get_response(prompt_name=None,model = model,custom_prompt = None,input_data = None):
    '''
    Description:
        Function to get response from Vertexai-Gemini Model
    Args:
        prompt_name : str : Name of the prompt
        model : GenerativeModel(optional) : Model object(default is vertexai-gemini model)
        Custom_prompt : str : Custom prompt
    Returns:
        response : str : Response from the model
    '''
    if custom_prompt:
        prompt = custom_prompt
    elif prompt_name:
        prompt = prompt_manager.get_model('vertexai').get_prompt(prompt_name,context = input_data)
    else:
        raise ValueError("Please provide either prompt_name or custom_prompt")   
    response = model.generate_content(prompt)
    return response.text