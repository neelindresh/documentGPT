from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain_core.documents.base import Document
from langchain_community.chat_models import AzureChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dataclasses import asdict
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import os
import json

from utils import pdf_utils
from config import OpenAIConfig, ChromaClient

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
    
class ConvertToVector:
    def __init__(self,embeddings,azure_forms) -> None:
        self.embeddings=Embeddings(embeddings).load()
        self.azure_forms=azure_forms
        
    def convert_to_vector(self,path,path_tovector_store,store_name):
        scanned_flag,_=pdf_utils.check_if_scanned_full_doc(path=path)
        if scanned_flag:
            print("Entered Into Scanned")
            file_names=pdf_utils.convert_to_doc_intell_pdf_format(path)
            docs=[]

            for idx,doc in enumerate(file_names):
                data=self.azure_forms.pdf_formatter(doc,original_path=path)
                docs.extend(data)
                
        else:
            docs=pdf_utils.process(path)
        
        document_format=pdf_utils.convert_to_langchain_docs(docs)
        vdb_path=os.path.join(path_tovector_store,store_name)
        if os.path.exists(vdb_path):
            print("ENter Here")
            vdb=Chroma(embedding_function=self.embeddings,persist_directory=vdb_path)
            vdb=vdb.from_documents(document_format,embedding=self.embeddings,persist_directory=vdb_path)
            
        else:
            vdb=Chroma.from_documents(document_format,embedding=self.embeddings,persist_directory=vdb_path)
        vdb.persist()
        
        
class LLMmodel:
    def __init__(self,embeddings,db_name) -> None:
        self.embeddings=Embeddings(embeddings).load()
        self._set_llm()
        self.vectordb=Chroma(embedding_function=self.embeddings,persist_directory=db_name)
        self.retriver=self.vectordb.as_retriever()
        self.chat_history=ChatMessageHistory()
        template = """Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Use three sentences maximum and keep the answer as concise as possible.
            Always say "thanks for asking!" at the end of the answer.

            {context}

            Question: {question}

            Helpful Answer:"""
        self.custom_rag_prompt = PromptTemplate.from_template(template)
        self.rag_chain=self.custom_rag_prompt | self.llm
    def _set_llm(self,params=None):
        configarations=asdict(OpenAIConfig())
        if params:
            configarations.update(params)
        self.llm=AzureChatOpenAI(**configarations)
        
    def predict(self,query):
        self.chat_history.add_user_message(query)
        data=self.retriver.invoke(query)
        context="\n\n".join([d.page_content for d in data])
        info_list=[d.metadata for d in data]
        responce=self.rag_chain.invoke({"question":query,"context":context})
        self.chat_history.add_ai_message(responce)
        
        return {
            "responce":responce.content,
            "info":[{"page":m['page'],"path":m["path"].split("/")[-1]} for m in info_list]
        }
        
        
class LLMmodelV1:
    def __init__(self,embeddings,db_name) -> None:
        self.embeddings=Embeddings(embeddings).load()
        self.last_idx=db_name
        self._set_llm()
        
        self._set_vdb(db_name)
        
        
    def _set_llm(self,params=None):
        configarations=asdict(OpenAIConfig())
        if params:
            configarations.update(params)
        self.llm=AzureChatOpenAI(**configarations)

    def _set_chat_history(self):
        self.chat_history=ChatMessageHistory()
        system_prompt='''Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Use three sentences maximum and keep the answer as concise as possible.
            '''
        self.custom_rag_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    system_prompt,
                ),
                MessagesPlaceholder(variable_name="messages"),
                
            ]
        )
        self.rag_chain=self.custom_rag_prompt | self.llm
        
    def _set_vdb(self,name):
        #if name!=self.last_idx:
        self.vectordb=Chroma(embedding_function=self.embeddings,persist_directory=name)
        self.retriver=self.vectordb.as_retriever(search_kwargs={'k':10})
        self.last_idx=name
        self._set_chat_history()
            
        
    
    def predict(self,query):
        
        data=self.retriver.invoke(query)
        context="\n\n".join([d.page_content for d in data])
        info_list=[d.metadata for d in data]
        unique = dict()
        for item in info_list:
            # concatenate key
            key = f"{item['path']}{item['page']}"
            # only add the value to the dictionary if we do not already have an item with this key
            if not key in unique:
                unique[key] = item
        info_list=list(unique.values())
                
        #format Question
        query_formatted=f'''
        {context}
        
        Question: {query}

        Helpful Answer:
        '''
        
        
        self.chat_history.add_user_message(query_formatted)
        responce=self.rag_chain.invoke({"messages": self.chat_history.messages})
        self.chat_history.add_ai_message(responce)
        template=f'Given the context \n{context} \n and Question: {query} \n Responce {responce.content}. Give me 3 related questions on this'
        followup_qa=self.llm.invoke(template)
        
        pdf_name_mapping={'az1742-2018.pdf':'Solar Photovoltic (PV) System Components.pdf',
                          '6981.pdf':"Photovoltics: Basic Design Princicals and Components.pdf",
                          'BOOK3.pdf':"Solar Photovoltics Technology and Systems.pdf"
                          }
        info_list=[{"page":m['page'],"path":m["path"].split("/")[-1]} for m in info_list]
        for idx,s in enumerate(info_list):
            if s["path"] in pdf_name_mapping:
                info_list[idx]["path"]=pdf_name_mapping[s["path"]]
                
        print(info_list)
        return {
            "output":responce.content,
            "metadata":{
                "sources":info_list,
                
            },
            "followup":followup_qa.content.split('\n')
        }
        



class AzureDocIntell:
    def __init__(self,api_key,end_point):
        credential = AzureKeyCredential(api_key)
        self.document_analysis_client = DocumentAnalysisClient(end_point, credential)
        
    def pdf_formatter(self,pdf,original_path):
        print(pdf)
        with open(pdf, "rb") as f:
            poller = self.document_analysis_client.begin_analyze_document(
                "prebuilt-layout", document=f
            )
        result = poller.result()
        data=[]
        for page in result.paragraphs:
            try:
                _temp={}
                _temp['block']=page.content
                _temp['page_no']=[b.page_number for b in page.bounding_regions][0]
                _temp['doc_name']=original_path
                data.append(_temp)
            except Exception as e:
                print(e)
                
        return data
    
    
    
class CompartiveAnalysis:

    def __init__(self,embeddings,db_name) -> None:
        self.embeddings=Embeddings(embeddings).load()
        self.last_idx=db_name
        self._set_llm()
        self.data_sources={"SAIL":"docs/SAIL Transcript Q3 FY24.pdf","tata":"docs/TATA STEELS 3qfy24-transcript-v7.pdf"}
        self.client = chromadb.HttpClient(**asdict(ChromaClient()))
        self.chat_history=ChatMessageHistory()

    def _set_llm(self,params=None):
        configarations=asdict(OpenAIConfig())
        if params:
            configarations.update(params)
        self.llm=AzureChatOpenAI(**configarations)
        
    def do_search(self,query):
        all_context=[]
        context_by_file={}
        #Serch and compare
        for doc in self.data_sources:

            vdb=Chroma(embedding_function=self.embeddings,persist_directory=self.last_idx,collection_name=doc,client=self.client)
            result=vdb.similarity_search(query,k=5,)
            all_context.extend(result)
            context_by_file[doc]=result
        return all_context
    
    def make_context(self,results):
        context=""
        company_name=""
        for r in results:
            context+="DocInfo: "+json.dumps(r.metadata)+ "\n"
            context+="DocContent: "+r.page_content+ "\n-------------\n"
            if 'Name/Company' in r.metadata:
                company_name=r.metadata['Name/Company']
            
        return context,[{"path":r.metadata['source'],"page":r.metadata['page']} for r in results]


    def info_extractor(self,context,query,chathistory):
        prompt=PromptTemplate.from_template("""
        Use the following pieces of context to answer the question at the end. Give full details about th topic
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Information is provided in the following format 
        DocInfo : General Information on the document
        DocContent: Actual data from the document

        NOTE: 
        - Please only Use the context provided to you.
        - Do a Comparison between diffent company
        
        
        
        Context:
        {doc}

        Chat History:
        {chathistory}
        
        query:
        {query}
        """,
        )

        chain = prompt | self.llm
        out=chain.invoke({"doc":context,"query":query,"chathistory":chathistory})
        return out.content
    def convert_to_string(self,history,n=2):
        text=""
        for m in history.dict()['messages'][-n:]:
            text+=f"{m['type']} : {m['content']}\n"
        return text
    
    def predict(self,query):
        
        all_context=self.do_search(query)
        print(all_context)
        context,info_list=self.make_context(all_context)
        history=self.convert_to_string(self.chat_history)
        out=self.info_extractor(context,query,history)
        self.chat_history.add_user_message(query)
        self.chat_history.add_ai_message(out)
        unique = dict()
        for item in info_list:
            # concatenate key
            key = f"{item['path']}{item['page']}"
            # only add the value to the dictionary if we do not already have an item with this key
            if not key in unique:
                unique[key] = item
        info_list=list(unique.values())
        info_list=[{"page":m['page'],"path":m["path"].split("/")[-1]} for m in info_list]
        template=f'Given the context \n{context} \n and Question: {query} \n Responce {out}. Give me 3 related questions on this'
        followup_qa=self.llm.invoke(template)
        return {
            "output":out,
            "metadata":{
                "sources":info_list,
                
            },
            "followup":followup_qa.content.split('\n')
        }