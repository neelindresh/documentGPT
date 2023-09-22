import openai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('api_key')
#Responce
deployment_name="LH-GPT"
api_type = "azure"
api_version = "2023-06-01-preview" 
api_base = "https://openai-lh.openai.azure.com/"  # Your Azure OpenAI resource's endpoint value.


#Embeddings
EMB_DEPLOYMENT_NAME="LH-embedding"
EMB_MODEL="text-embedding-ada-002"


#Hugginface_T5_Model
HUGGINFACE_MODEL=None
HUGGINFACE_MODEL_OFFLOAD=None
HUGGINFACE_MODEL_PARAMS={}


BASE_PATH=os.getcwd()
MODEL_FOLDER="models"
EMD_FOLDER="E5-base-v2"
EMD_PATH=os.path.join(BASE_PATH,MODEL_FOLDER,EMD_FOLDER)
#EMD_PATH="intfloat/e5-base-v2"
DATA_FOLDER="data"
DATA_PATH=os.path.join(BASE_PATH,DATA_FOLDER)
