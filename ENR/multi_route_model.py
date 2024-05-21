from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
from dataclasses import dataclass,asdict
from langchain_community.chat_models import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
import pandas as pd
from langchain.memory import ChatMessageHistory
from config import ChromaClient,OpenAIConfig,ChromaClientDEV,OpenAI4
from ENR.multi_route_model_chains import router_chain,metafinder,regulation_chat,tender_chat, tender_regulation_miner,get_agent_brain

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
    
    
class ENR_multiroute_Chat:
    def __init__(self) -> None:
        model_name="mixedbread-ai/mxbai-embed-large-v1"
        self.azureopenai=AzureChatOpenAI(**asdict(OpenAIConfig()))
        self.azureopenai_gpt_4=AzureChatOpenAI(**asdict(OpenAI4()))
        self.chat_history=ChatMessageHistory()
        self.embeddings=Embeddings(model_name).load()
        self.client= chromadb.HttpClient(**asdict(ChromaClientDEV()))
        self.router=router_chain(self.azureopenai)
        self.meta_filter_chain=metafinder(self.azureopenai)
        self.regulation_chain=regulation_chat(self.azureopenai)
        self.tender_chain=tender_chat(self.azureopenai)
        self.tandrchain=tender_regulation_miner(self.azureopenai)
        self.brain_chain=get_agent_brain(self.azureopenai_gpt_4)
        with open("ENR/summaryindex.txt","r") as f:
            self.TENDER_SUMMARY_INDEX=f.read()
        self.last_info=[]
        self.VECTOR_DBS={
            "tender":"ENR_TENDERS_WITH_AMMENDMENTS",
            "regulation":['REGULATIONS_1', 'REGULATIONS_2', 'REGULATIONS_3', 'REGULATIONS_4'],
            "summary":"ENR_TENDER_SUMMARY"
        }
        self.meta_json=[{'regulation_name': 'CERC/SERC regulations on Forecasting, Scheduling and Deviation Settlement',
            'subject': "Approval of the 'Procedure for implementation of the Framework on Forecasting, Scheduling and Imbalance Handling for Renewable Energy (RE) Generating Stations including Power Parks based on Wind and Solar at Inter-State level'",
            'topics': ['Electricity Regulation',
            'Renewable Energy',
            'Forecasting',
            'Scheduling',
            'Imbalance Handling',
            'Wind Energy',
            'Solar Energy'],
            'filename': 'CERC Regulations - Forecasting, Scheduling and Deviation Settlement.pdf',
            'collection_name': 'REGULATIONS_1'},
            {'regulation_name': 'Central Electricity Regulatory Commission (Connectivity and General Network Access to the inter-State Transmission System) Regulations, 2022',
            'subject': 'Regulatory framework for non-discriminatory open access to the inter-State transmission system through General Network Access',
            'topics': ['Electricity Act',
            'Transmission System',
            'Regulatory Framework',
            'Open Access',
            'Connectivity',
            'General Network Access'],
            'filename': 'GNA-175.pdf',
            'collection_name': 'REGULATIONS_2'},
            {'regulation_name': 'Grid Code',
            'subject': 'Specifications and regulations for the operation and maintenance of the Grid and its components, including Grid Standards, Grid Code, and the roles of the Central Electricity Authority and Regional Load Despatch Centres.',
            'topics': ['Electricity Act, 2003',
            'Central Electricity Regulatory Commission',
            'Grid Standards',
            'Grid Code',
            'National Load Despatch Centre',
            'Regional Load Despatch Centre'],
            'filename': 'Grid-Code- 180.pdf',
            'collection_name': 'REGULATIONS_3'},
            {'regulation_name': 'विद्युत मंत्रालय संकल्प',
            'subject': 'ऊर्जा भंडारण प्रणालियों सहित ग्रिड संबद्ध नवीकरणीय ऊर्जा विद्युत परियोजनाओं से स्थिर तथा प्रेषणयोग्य विद्युत की खरीद हेतु टैरिफ आधारित प्रतिस्पर्धी बोली प्रक्रिया के लिए दिशानिर्देश।',
            'topics': ['Energy Storage Systems',
            'Grid-Connected Renewable Energy',
            'Electricity Purchase',
            'Competitive Bidding',
            'Tariff-Based'],
            'filename': 'Guidelines_for_Tariff_Based_Competitive_Bidding_Process_for_Procurement.pdf',
            'collection_name': 'REGULATIONS_4'}]
                
    def get_pipeline(self,query):
        
        route=self.router.invoke({"query":query})
        if route.content.startswith("Output:"):
            return route.content.strip("Output:").strip().split(",")
        else:
            return route.content.strip().split(",")
    def get_tenderlist(self,query):
        tenders=pd.DataFrame([{"Tender Name": "RFS 1000 MW","Status":"Open","Tender By":"SECI"},
                {"Tender Name": "RFS 1250 MW","Status":"Open","Tender By":"SECI"},
                {"Tender Name": "RFS 2000 MW","Status":"Closed","Tender By":"SECI"},
                ]).to_markdown()
        return tenders,[]
    
    def get_regulations(self,query):
        
        meta_filters=self.meta_filter_chain.invoke({"metastore":json.dumps(self.meta_json),"query":query})
        collections=eval(meta_filters.content)
        
        all_res=[]
        for c in collections:
            regulation_db=Chroma(embedding_function=self.embeddings,client=self.client,persist_directory="VectorDB",collection_name=c)
            regulation_retriver=regulation_db.as_retriever()
            res=regulation_retriver.invoke(query)
            all_res.extend(res)
        
        
        context="\n  ".join([d.page_content for d in all_res])
        info_list=[d.metadata for d in all_res if d.metadata['page']!=""]
        info_list=self.convert_to_sources(info_list)
        
        out=self.regulation_chain.invoke({
                    "query": query,
                    "context": context,
                })
        return out.content,info_list
    def get_tender(self,query):
        tender_db=Chroma(embedding_function=self.embeddings,client=self.client,persist_directory="VectorDB",collection_name=self.VECTOR_DBS['tender'])
        tender_retriver=tender_db.as_retriever()
        summary_db=Chroma(embedding_function=self.embeddings,client=self.client,persist_directory="VectorDB",collection_name=self.VECTOR_DBS['summary'])
        summary_retriver=summary_db.as_retriever()
        
        res=tender_retriver.invoke(query)
        sumres=summary_retriver.invoke(query)
        info_list=[d.metadata for d in res if d.metadata['page']!=""]
        info_list=self.convert_to_sources(info_list)
        context="\n  ".join([d.page_content for d in res])
        out=self.tender_chain.invoke({
                    "query": query,
                    "context": context,
                    "summary": "\n  ".join([d.page_content for d in sumres]),
                })
        return out.content,info_list
    def tender_and_regulation(self,query):
        tender_ans=self.get_tender(query)
        regulations=self.get_regulations(query)
        #print("Tender",tender_ans)
        #print("---------------------")
        #print("Regulation",regulations)
        #print("---------------------")
        
        out=self.tandrchain.invoke({
            "query": query,
            "regulation":regulations,
            "tender": tender_ans
        })
        return out.content
    
    
    def get_summary(self,query):
        print("IS FULL?")
        template=f'''Does the user want a index or summary of the whole document, or all sections of the document?
        Examples:
        --------------
        Input: Can you give me a section wise summary of Tender 0
        Output: Yes
        
        Input: Can you give me a section wise summary of Tender 0
        Output: Yes
        
        Input: Can you summarize the Sections of the rfs
        Output: yes

        Input: Can you give the list of sections in the document?
        Output: yes
        
        Input: Can you give me a summary of Section 7.4 from Tender 0?
        Output: No
        
        Input: Can you give me a summary of QUALIFICATION REQUIRMENTS FOR BIDDER
        Output: No
        
        Input: Can you give me a summary of how the project should be designed for interconnection with the ISTS
        Output: No
        
        
        user query: {query}
        Answer in only 'Yes' or 'No'
        '''
        answer=self.azureopenai.invoke(template)
        print("-------->",answer.content)
        if answer.content.lower()=="yes":
            return self.TENDER_SUMMARY_INDEX
        else:
            summary_db=Chroma(embedding_function=self.embeddings,client=self.client,persist_directory="VectorDB",collection_name=self.VECTOR_DBS['summary'])
            summary_retriver=summary_db.as_retriever()
            
            sumres=summary_retriver.invoke(query)
            info_list=[d.metadata for d in sumres if d.metadata['page']!=""]
            info_list=self.convert_to_sources(info_list)
            context="\n  ".join([d.page_content for d in sumres])
            out=self.regulation_chain.invoke({
                    "query": query,
                    "context": context,
            })
            return out.content,info_list
        
    def convert_to_string(self,history,n=2):
        text=""
        for m in history.dict()['messages'][-n:]:
            text+=f"{m['type']} : {m['content']}\n"
        return text
    def convert_to_sources(self,info_list):
        unique = dict()
        for item in info_list:
            # concatenate key
            key = f"{item['path']}{item['page']}"
            # only add the value to the dictionary if we do not already have an item with this key
            if not key in unique:
                unique[key] = item
        info_list=list(unique.values())
        info_list=[{"page":m['page'],"path":m["path"].split("/")[-1]} for m in info_list]
        return info_list
    def get_agent_chain(self,query,chat_history):
        
        out=self.brain_chain.invoke(
        {
            "query":query,
            "chat_history":self.convert_to_string(chat_history)
        })
        return out.content
    def predict(self,query):
        print("Enter Brain Agent")
        self.brain_responce=self.get_agent_chain(query,self.chat_history)
        print("---->",self.brain_responce)
        if self.brain_responce.lower()=='no':
            print("Enter Router Agent")
            route=self.get_pipeline(query)
            print("ROUTER-->",route)
            if "Tender" in route and "Regulation" in route:
                responce,info_list=self.tender_and_regulation(query)
            elif "Tender" in route:
                responce,info_list=self.get_tender(query)
            elif "Regulation" in route:
                responce,info_list=self.get_regulations(query)
            elif "Summary" in route:
                responce,info_list=self.get_summary(query)
            elif "OpenTender" in route:
                responce,info_list=self.get_tenderlist(query)
            else:
                responce="Can you provide more information ?"
                info_list=[]
            self.chat_history.add_user_message(query)
            self.chat_history.add_ai_message(responce)
            self.last_info=info_list
            template=f'Given the  Question: {query} \n Responce {responce}. Give me 3 related questions on this'
            followup_qa=self.azureopenai.invoke(template)
            return {
                "output":responce,
                "metadata":{
                    "sources":info_list,
                    
                },
                "followup":followup_qa.content.split('\n')
            }
            
        else:
            self.chat_history.add_user_message(query)
            self.chat_history.add_ai_message(self.brain_responce)
            template=f'Given the  Question: {query} \n Responce {responce}. Give me 3 related questions on this'
            followup_qa=self.azureopenai.invoke(template)
            return  {
                "output":self.brain_responce,
                "metadata":{
                    "sources":self.last_info,
                    
                },
                "followup":followup_qa.content.split('\n')
            }