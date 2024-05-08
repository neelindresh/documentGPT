from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
from dataclasses import dataclass,asdict
from langchain_community.chat_models import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
import pandas as pd
from langchain.memory import ChatMessageHistory
from config import ChromaClient,OpenAIConfig
from ENR.ENR_templates import regclassificationChatTemplate,chatTemplate,regulationChatTemplate,VECTOR_DBS
from ENR.temp_data_model import tendersummary,reg_mapping
'''
__import__('pysqlite3')
import sys
sys.modules['sqlite3']= sys.modules.pop('pysqlite3')
'''

import chromadb
from chromadb.config import Settings

class Embeddings:
    def __init__(self,name) -> None:
        self.name=name
    def load(self):
        return HuggingFaceEmbeddings(model_name=self.name)

class ENR_Chat:
    def __init__(self) -> None:
        model_name="mixedbread-ai/mxbai-embed-large-v1"
        self.azureopenai=AzureChatOpenAI(**asdict(OpenAIConfig()))
        self.chat_history=ChatMessageHistory()
        self.embeddings=Embeddings(model_name).load()
        self.client= chromadb.HttpClient(**asdict(ChromaClient()))
        self.class_chain=regclassificationChatTemplate|self.azureopenai
        self.tender_chain=chatTemplate|self.azureopenai
        self.regulation_chain=regulationChatTemplate |self.azureopenai
        
        tender_db=Chroma(embedding_function=self.embeddings,client=self.client,persist_directory="VectorDB",collection_name=VECTOR_DBS['tender'])
        self.tender_db=tender_db.as_retriever()
        regulation_db=Chroma(embedding_function=self.embeddings,client=self.client,persist_directory="VectorDB",collection_name=VECTOR_DBS['regulation'])
        self.regulation_db=regulation_db.as_retriever()
        summary_db=Chroma(embedding_function=self.embeddings,client=self.client,persist_directory="VectorDB",collection_name=VECTOR_DBS['summary'])
        self.summary_db=summary_db.as_retriever()
    def convert_to_string(self,history,n=2):
        text=""
        for m in history.dict()['messages'][-n:]:
            text+=f"{m['type']} : {m['content']}\n"
        return text
    def predict(self,query):
        self.chat_history.add_user_message(query)
        classification=self.class_chain.invoke({"query":query,"reg_mapping":reg_mapping,"tender_summary":tendersummary})
        classPred=True
        if classification.content=="Tender":
            print("Tender")
            #query="can you tell me the list regulations are mentioned"
            res=self.tender_db.invoke(query)
            sumres=self.summary_db.invoke(query)
            context="\n  ".join([d.page_content for d in res])
            info_list=[d.metadata for d in res if d.metadata['page']!=""]
            out=self.tender_chain.invoke({
                "query": query,
                "context": context,
                "summary": "\n  ".join([d.page_content for d in sumres]),
                "chat_history": self.convert_to_string(self.chat_history)
            })
        elif classification.content=="Regulation":
            print("Regulation")
            reg_results=self.regulation_db.invoke(query)
            context="\n  ".join([d.page_content for d in reg_results])
            info_list=[{"path":d.metadata['source'],"page":d.metadata['page']} for d in reg_results if d.metadata['page']!=""]
            out=self.regulation_chain.invoke({
                "query": query,
                "context": context,
                "chat_history": self.convert_to_string(self.chat_history)
            })
        else:
            classPred=False
        if classPred:
            self.chat_history.add_ai_message(out.content)
            template=f'Given the context \n{context} \n and Question: {query} \n Responce {out}. Give me 3 related questions on this'
            followup_qa=self.azureopenai.invoke(template)
            unique = dict()
            for item in info_list:
                # concatenate key
                print(item)
                key = f"{item['path']}{item['page']}"
                # only add the value to the dictionary if we do not already have an item with this key
                if not key in unique:
                    unique[key] = item
            info_list=list(unique.values())
            info_list=[{"page":m['page'],"path":m["path"].split("/")[-1]} for m in info_list]
            
            return {
                "output":out.content,
                "metadata":{
                    "sources":info_list,
                    
                },
                "followup":followup_qa.content.split('\n')
            }
        else:
            return {
                "output":"Server is down due to high traffic. Please try after a few seconds",
                "metadata":{
                    "sources":[],
                    
                },
                "followup":""
            }