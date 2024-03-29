from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain_core.documents.base import Document
from langchain_community.chat_models import AzureChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dataclasses import asdict

from utils import pdf_utils
from config import OpenAIConfig
'''
__import__('pysqlite3')
import sys
sys.modules['pysqlite3']= sys.modules.pop('pysqlite3')
'''

class Embeddings:
    def __init__(self,name) -> None:
        self.name=name
    def load(self):
        return HuggingFaceEmbeddings(model_name=self.name)
    
class ConvertToVector:
    def __init__(self,embeddings) -> None:
        self.embeddings=Embeddings(embeddings).load()
        
    def convert_to_vector(self,path,store_path):
        docs=pdf_utils.process(path)
        document_format=pdf_utils.convert_to_langchain_docs(docs)
        vdb=Chroma.from_documents(document_format,embedding=self.embeddings,persist_directory=store_path.strip('.pdf'))
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
            "info":info_list
        }
        
        
class LLMmodelV1:
    def __init__(self,embeddings,db_name) -> None:
        self.embeddings=Embeddings(embeddings).load()
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
        self.vectordb=Chroma(embedding_function=self.embeddings,persist_directory=name)
        self.retriver=self.vectordb.as_retriever()
        
        self._set_chat_history()
        
    
    def predict(self,query):
        
        data=self.retriver.invoke(query)
        context="\n\n".join([d.page_content for d in data])
        info_list=[d.metadata for d in data]
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
        return {
            "responce":responce.content,
            "info":info_list,
            "followup":followup_qa.content.split('\n')
        }