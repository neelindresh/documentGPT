from langchain_openai.chat_models import AzureChatOpenAI
from dataclasses import dataclass,asdict
import chromadb
from chromadb.config import Settings
from langchain.memory import ChatMessageHistory



from config import OpenAIConfig,ChromaClientDEV
from utils.model_utils import Embeddings
from AdvanceRag.advanced_rag import advanced_retrival
from AdvanceRag.prompt_utils import get_rag

class CompartiveAnalysisAdvancedRag:
    def __init__(self,emd_name) -> None:
        self.llm=AzureChatOpenAI(**asdict(OpenAIConfig()))
        self.embeddings=Embeddings(emd_name).load()
        self.client=self.client= chromadb.HttpClient(**asdict(ChromaClientDEV()))
        self.rag_chain=get_rag(self.llm)
        self.chat_history=ChatMessageHistory()
        ## CODE- Fetch Metastore from mongoDB, usecase collection
        self.meta_store=[{'organization': 'TATA STEEL',
            'period': 'November 01, 2023',
            'subject': 'Tata Steel reports Consolidated EBITDA of Rs 4,315 crores for the quarter',
            'topics': [],
            'filename': '2qfy24-press-release.pdf',
            'collection_name': 'TSVSJSW_0'},
            {'organization': 'JSW Steel Limited',
            'period': 'Third Quarter FY 2023-24',
            'subject': 'Financial Performance',
            'topics': [],
            'filename': 'JSW Press-Release-Q3-FY24.pdf',
            'collection_name': 'TSVSJSW_1'},
            {'organization': 'JSW Steel',
            'period': 'Second Quarter FY 2023-24',
            'subject': 'Financial Performance',
            'topics': [],
            'filename': 'Press-Release-Q2FY-24.pdf',
            'collection_name': 'TSVSJSW_2'},
            {'organization': 'TATA STEEL',
            'period': 'January 24, 2024',
            'subject': 'NEWS RELEASE',
            'topics': [],
            'filename': 'TSL 3qfy24-press-release.pdf',
            'collection_name': 'TSVSJSW_3'}]
    def convert_to_string(self,history,n=2):
        text=""
        for m in history.dict()['messages'][-n:]:
            text+=f"{m['type']} : {m['content']}\n"
        return text
    
    def predict(self,query):
        prev_conv=self.convert_to_string(self.chat_history)
        data=advanced_retrival(self.llm,self.meta_store,query=query,embeddings=self.embeddings,chroma_client=self.client,prev_conv=prev_conv)
        context="\n\n".join([d.page_content for d in data])
        out=self.rag_chain.invoke({
            "context":context,
            "user_query":query,
            "chat_history":prev_conv
        })
        self.chat_history.add_user_message(query)
        self.chat_history.add_ai_message(out.content)
        #Same code
        info_list=[d.metadata for d in data]
        unique = dict()
        for item in info_list:
            # concatenate key
            key = f"{item['path']}{item['page']}"
            # only add the value to the dictionary if we do not already have an item with this key
            if not key in unique:
                unique[key] = item
        info_list=list(unique.values())
        info_list=[{"page":m['page'],"path":m["path"].split("/")[-1]} for m in info_list]
        template=f'Given the context \n{context} \n and Question: {query} \n Responce {out.content}. Give me 3 related questions on this'
        followup_qa=self.llm.invoke(template)
        return {
            "output":out.content,
            "metadata":{
                "sources":info_list,
                
            },
            "followup":followup_qa.content.split('\n')
        }
        
        