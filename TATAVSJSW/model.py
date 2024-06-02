from TATAVSJSW.utils import load_embeddings,load_reranker,ensemble_retriver,rerank_docs,get_document_source,get_final_documents,MAX_TOKEN_CONTEXT_LIMIT

import chromadb
from chromadb.config import Settings
from config import ChromaClientDEV,UseCaseMongo,OpenAI4,OpenAIConfig
from langchain_community.chat_models import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ChatMessageHistory


from dataclasses import asdict
import numpy as np
from utils.mongoutils import MongoConnect
from TATAVSJSW.index import RETURN_INDEX
MAX_TOKEN_CONTEXT_LIMIT=5000

def router_chain(llm):

    prompt='''
    Given a user query, Classify it into Either `Index` or `QA`

    // Index: If the query is about a summarizing the document, or its content
    // QA: Other questions Related to a specific organizarion, or request to provide data in a specific format


    Examples:
    
    Input: Give me the comparative summary of the KPIs for Tata Steel and JSW
    Output: Index
    
    Input: Give me the comparative View of the All the KPIs for Tata Steel and JSW
    Output: Index
    
    Input: can you please provide an in depth analysis on the crude steel capacity of tata steel and their further expansion plans
    Output: QA
    
    Input: can you please provide an in depth analysis of the M.Cap. of tata steel?
    Output: QA

    Input: can you please provide an in depth analysis on the crude steel capacity of tata steel and their further expansion plans
    Output: QA

    Input: can you please provide an in depth analysis on the market share of tata steel and their breakup in domestic and international sales
    Output: QA

    Input: can you provide the start date and the expected completion date for the capex projects of Tata Steel
    Output: QA

    Input: Can you provide the EBITDA of Tata Steel for different quarters(Q) and fiscal years (FY)?
    Output: QA

    Input: Can you provide insights on the sales of JSW steel in tons?
    Output: QA

    Input: Can you provide insights on the sales growth of JSW steel?
    Output: QA

    Input: Can you provide insights on JSW steels plans and planned initiatives for growing their sales?
    Output: QA

    Input: What are the primary factors driving revenue growth for Tata Steel for FY2022-23 and FY2021-22?What is the reason behind revenue growth for JSW Steel for FY2022-23 and FY2021-22?Can you provide a comparison of the reasons for revenue growth of JSW Steel and Tata steel for FY2022-23 and FY2021-22?
    Output: QA

    Input: How is the financial performance and trends of Tata steel as compared to JSW? Give the answer in a table format
    Output: QA

    Input: What are the primary factors driving revenue growth for Tata Steel for FY2022-23 and FY2021-22?
    Output: QA
    
    
    Input: {query}
    Output: 
    '''

    routerChatTemplate=PromptTemplate(
        input_variables=["query"],
        template=prompt,
    )
    reg_class_chain=routerChatTemplate | llm
    return reg_class_chain


def rag_chat(llm):
    tender_chat_prompt='''You are a helpful assitant who is master in tender analysis.

    Given the content, please provide the answer to the user query as good as possible. Please provide as much details as possible

    Chat History
    ---------
    {chat_history}

    Context
    ---------
    {context}

    User Query
    ---------
    {query}
    '''

    chatTemplate=PromptTemplate(
        input_variables=["query","context","chat_history"],
        template=tender_chat_prompt,
    )
    return chatTemplate | llm

class TATAVSJSWModel:
    def __init__(self,usecase_id) -> None:
        self.embeddings=load_embeddings()
        self.reranker=load_reranker()
        self.mongo_client=MongoConnect(**asdict(UseCaseMongo()))
        self.client = chromadb.HttpClient(**asdict(ChromaClientDEV()))
        self.data_source=self.mongo_client.get_data_by_id(usecase_id)#"1717087491223")
        self.azureopenai=AzureChatOpenAI(**asdict(OpenAIConfig()))
        self.azureopenai_gpt_4=AzureChatOpenAI(**asdict(OpenAI4()))
        self.router=router_chain(self.azureopenai)
        self.rag_chain=rag_chat(self.azureopenai)
        self.chat_history=ChatMessageHistory()

    
    def advance_retrival(self,query,data_sources):
        if 'meta_data' not in data_sources:
            raise Exception("'meta_data' feild missing in data sources")
        #collections=[c['collection_name'] for c in data_sources['meta_data']]
        meta_data=data_sources['meta_data']
        #embeddings=load_embeddings()
        #reranker=load_reranker()
        
        retrived_documents=ensemble_retriver(query,meta_data,self.client,self.embeddings)
        document_score=rerank_docs(query,retrived_documents,self.reranker)
        
        sorted_arg_idx=list(np.argsort(document_score))[::-1]
        final_docs=get_final_documents(sorted_arg_idx,document_score,retrived_documents,MAX_TOKEN_CONTEXT_LIMIT)
        context="\n\n".join([i.page_content for i in final_docs])
        info_list=get_document_source(final_docs)
        return {
            "context": context,
            "info_list":info_list
        }
    def get_pipeline(self,query):
        
        route=self.router.invoke({"query":query})
        if route.content.startswith("Output:"):
            return route.content.strip("Output:").strip().split(",")
        else:
            return route.content.strip().split(",")

    def convert_to_string(self,history,n=3):
        text=""
        for m in history.dict()['messages'][-n:]:
            text+=f"{m['type']} : {m['content']}\n"
        return text
    
    def predict(self,query):
        
        route=self.get_pipeline(query)[0]
        print(route)
        if route.lower()=="qa":
            output=self.advance_retrival(query,self.data_source["data_sources"])
            responce=self.rag_chain.invoke({
                "query":query,
                "context":output['context'],
                "chat_history":self.convert_to_string(self.chat_history)
                
            })
            context=output['context']
            self.chat_history.add_user_message(query)
            self.chat_history.add_ai_message(responce.content)
            template=f'Given the context \n{context} \n and Question: {query} \n Responce {responce.content}. Give me 3 related questions on this'
            followup_qa=self.azureopenai.invoke(template)
            return  {
                "output":responce.content,
                "metadata":{
                    "sources":output['info_list'],
                    
                },
                "followup":followup_qa.content.split('\n')
            }
        elif route.lower()=="index":
            context=RETURN_INDEX
            template=f'Given the context \n{context[:-10000]} \n and Question: {query} . Give me 3 related questions on this'
            followup_qa=self.azureopenai.invoke(template)
            return {
                "output":context,
                "metadata":{
                    "sources":[],
                    
                },
                "followup":followup_qa.content.split('\n')
            }
        else:
            return {
                "output":"Can you please Rephrase the Question?",
                "metadata":{
                    "sources":[],
                    
                },
                "followup":followup_qa.content.split('\n')
            }
        
        
        