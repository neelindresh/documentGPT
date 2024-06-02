from dataclasses import dataclass,asdict

@dataclass
class OpenAIConfig:
    api_key:str="312ff50d6d954023b8748232617327b6"
    azure_endpoint:str="https://openai-lh.openai.azure.com/"
    azure_deployment:str="test"
    api_version:str="2024-02-15-preview"

@dataclass
class OpenAI4:
    azure_endpoint:str="https://lh-openai-4.openai.azure.com/"
    api_key:str="04030134596b4df3b7cceb3c77f5b0a9"
    azure_deployment:str="LH-GPT4"
    api_version:str="2024-02-01"


@dataclass
class AzureDocumentInfo:
    api_key:str='f8c8e2179f44484c872de1bd373c17c0'
    end_point:str='https://spendanalytics.cognitiveservices.azure.com/'
    
    
@dataclass
class ChromaClient:
    host:str="http://20.41.249.147:6062"
    port:int=8000
    
@dataclass
class ChromaClientDEV:
    host:str="http://52.172.103.119:6062"
    port:int=8000
    
    
@dataclass
class UseCaseMongo:
    uri:str = "mongodb+srv://ikegai:ikegai%40123456@cluster0.l2apier.mongodb.net"
    collection:str='usecases'
    db:str='ikegai_dev'