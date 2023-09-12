
from langchain.llms import AzureOpenAI
import configFolder.config as config
import configFolder.workflow as workflow

from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI

from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings,OpenAIEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA,ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFDirectoryLoader
import os

ACTIVE_WORKFLOW=workflow.Workflow


class Model:
    def __init__(self) -> None:
        embeddings = OpenAIEmbeddings(
         deployment="LH-embedding",
        model="text-embedding-ada-002",
        openai_api_type='azure',chunk_size=1300,chunk_overlap=100)

class AzureOpenAIServices:
    def __init__(self) -> None:
        self.llm = AzureOpenAI(
            openai_api_base=config.api_base,
            openai_api_version=config.api_version,
            deployment_name=config.deployment_name,
            openai_api_key=config.api_key,
            openai_api_type="azure",
            temperature=0.5
        )
        data=self._load_pdf()
        self._load_embedding()
        chunks=self._preprocessor(data)
        docsearch=self._init_db(chunks)
        template=self._prompt_template()
        self._qa(docsearch,template)
        
    def _load_pdf(self,path:str=None):
        if path==None:
            loader = PyMuPDFLoader("data/Access-Control-Standards_Latest.pdf")
        else:
            loader = PyMuPDFLoader(path)

        return loader.load()
    
    def _load_embedding(self):
        self.embedding=HuggingFaceEmbeddings(model_name=config.EMD_PATH)
        
    def _init_db(self,data):
        docsearch = Chroma.from_documents(data, self.embedding)
        return docsearch
    def _preprocessor(self,data):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=512,
            chunk_overlap=200,
            length_function=len,
        )
        text=""
        for p in data:
            text+=p.page_content
        chunks = text_splitter.split_documents(data)
        return chunks
    def _prompt_template(self):
        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": PROMPT}
        return chain_type_kwargs
    

    def _qa(self,docsearch,chain_type_kwargs):
        self.qa_chain = RetrievalQA.from_chain_type(self.llm,
                                                    retriever=docsearch.as_retriever(search_type="mmr"),
                                                        return_source_documents=True,
                                                        chain_type_kwargs = chain_type_kwargs
                                                    )

    def predict(self,query):
        results=self.qa_chain({"query": query})
        return results['result']


class ConversationAzureOpenAI:
    def __init__(self) -> None:
        self.llm = AzureOpenAI(
            openai_api_base=config.api_base,
            openai_api_version=config.api_version,
            deployment_name=config.deployment_name,
            openai_api_key=config.api_key,
            openai_api_type="azure",
            temperature=0.5
        )
        print("Loading Pdf")
        data=self._load_pdf()
        #print(data)
        print("Loading Embedings")
        self._load_embedding()
        print("Loading Preprocessor")
        chunks=self._preprocessor(data)
        print("Loading DB")
        docsearch=self._init_db(chunks)
        template=self._prompt_template()
        print("Loading Model")
        self._qa(docsearch,template)
        self.chat_history=[]
        
    def _load_pdf(self,path:str=None):
        if path==None:
            loader = PyMuPDFLoader("data/Access-Control-Standards_Latest.pdf")
        else:
            loader = PyMuPDFLoader(path)

        return loader.load()
    
    def _load_embedding(self):
        self.embedding=HuggingFaceEmbeddings(model_name=config.EMD_PATH)
        
    def _init_db(self,data):
        persist_directory = "./storage"

        docsearch = Chroma.from_documents(data, self.embedding,)#persist_directory=persist_directory)
        #docsearch.persist()
        #docsearch=Chroma( embedding_function=self.embedding,persist_directory=persist_directory)
        return docsearch
    def _preprocessor(self,data):
        text_splitter = RecursiveCharacterTextSplitter(
            
            chunk_size=512,
            chunk_overlap=200,
            length_function=len,
        )
        #text=""
        #for p in data:
        #    text+=p.page_content
        chunks = text_splitter.split_documents(data)
        return chunks
    def _prompt_template(self):
        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": PROMPT}
        return chain_type_kwargs
    

    def _qa(self,docsearch,chain_type_kwargs):
        '''
        self.qa_chain = RetrievalQA.from_chain_type(self.llm,
                                                    retriever=docsearch.as_retriever(search_type="mmr"),
                                                        return_source_documents=True,
                                                        chain_type_kwargs = chain_type_kwargs
                                                    )
        '''
        self.qa_chain=ConversationalRetrievalChain.from_llm(llm=self.llm,retriever=docsearch.as_retriever(search_type="mmr",search_kwargs={"k": 3}), return_source_documents=True,verbose=True)

    def predict(self,query):
        results=self.qa_chain({"question": query,"chat_history": self.chat_history})
        output=results['answer']
        self.chat_history.append((query,output))
        return output



class ConversationMultiDocumentAzureOpenAI:
    def __init__(self) -> None:
        self.llm = AzureOpenAI(
            openai_api_base=config.api_base,
            openai_api_version=config.api_version,
            deployment_name=config.deployment_name,
            openai_api_key=config.api_key,
            openai_api_type="azure",
            temperature=0.5
        )
        data=self._load_pdf()
        self._load_embedding()
        chunks=self._preprocessor(data)
        docsearch=self._init_db(chunks)
        template=self._prompt_template()
        self._qa(docsearch,template)
        self.chat_history=[]
        
    def _load_pdf(self,path:str=None):
        if path==None:
            loader = PyPDFDirectoryLoader("tata_steels_docs/")
        else:
            loader = PyPDFDirectoryLoader("tata_steels_docs/")

        return loader.load()
    
    def _load_embedding(self):
        self.embedding=HuggingFaceEmbeddings(model_name=config.EMD_PATH)
        
    def _init_db(self,data):
        persist_directory = "./storage_tata"

        docsearch = Chroma.from_documents(data, self.embedding,persist_directory=persist_directory)
        docsearch.persist()
        #docsearch=Chroma( embedding_function=self.embedding,persist_directory=persist_directory)
        return docsearch
    
    def _preprocessor(self,data):
        text_splitter = RecursiveCharacterTextSplitter(
            
            chunk_size=512,
            chunk_overlap=200,
            length_function=len,
        )
        #text=""
        #for p in data:
        #    text+=p.page_content
        chunks = text_splitter.split_documents(data)
        return chunks
    def _prompt_template(self):
        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": PROMPT}
        return chain_type_kwargs
    

    def _qa(self,docsearch,chain_type_kwargs):
        '''
        self.qa_chain = RetrievalQA.from_chain_type(self.llm,
                                                    retriever=docsearch.as_retriever(search_type="mmr"),
                                                        return_source_documents=True,
                                                        chain_type_kwargs = chain_type_kwargs
                                                    )
        '''
        self.qa_chain=ConversationalRetrievalChain.from_llm(llm=self.llm,retriever=docsearch.as_retriever(search_type="mmr",search_kwargs={"k": 3}), return_source_documents=True,verbose=True)

    def predict(self,query):
        results=self.qa_chain({"question": query,"chat_history": self.chat_history})
        info=[i.metadata for i in results["source_documents"]]
        output=results['answer']
        self.chat_history.append((query,output))
        return output,info



'''
def _load_model(self):
        self.tokenizer = T5Tokenizer.from_pretrained(config.LLM_MODEL)
        self.model = T5ForConditionalGeneration.from_pretrained(config.LLM_MODEL, device_map="auto",offload_folder=config.LLM_MODEL_OFFLOAD)
        self.pipe=pipeline('text2text-generation',model=self.model,tokenizer=self.tokenizer, **config.PARAMS)
def load_llm_model(self):
    return HuggingFacePipeline(pipeline=self.pipe)

'''



chat_history=[]
class MultiDocumentChatAzureOpenAI:
    def __init__(self) -> None:
        '''
        self.llm = AzureOpenAI(
            openai_api_base=config.api_base,
            openai_api_version=config.api_version,
            deployment_name=config.deployment_name,
            openai_api_key=config.api_key,
            openai_api_type="azure",
            temperature=0.5
        )
        '''
        self.llm = AzureChatOpenAI(
            openai_api_base=config.api_base,
            openai_api_version=config.api_version,
            deployment_name=config.deployment_name,
            openai_api_key=config.api_key,
        )
        
        self._load_embedding()
        persist_directory = ACTIVE_WORKFLOW["persist_directory"]
        if not os.path.exists(persist_directory):
            print("+++++++++++++++++++inside create new DB+++++++++++++++++")
            data=self._load_pdf_folder()
            #print(data)
            chunks=self._preprocessor(data)
            docsearch=self._init_db(chunks)
        docsearch=self._init_db_from_storage()
        template=self._prompt_template()
        self._qa(docsearch,template)
        self.chat_history=[]
        
    def _load_pdf_folder(self):
        
        if ACTIVE_WORKFLOW["folder"]==None:
            loader=PyMuPDFLoader(ACTIVE_WORKFLOW["pdf"])
        else:
            loader = PyPDFDirectoryLoader(ACTIVE_WORKFLOW["folder"])
        return loader.load()
    
    def _load_embedding(self):
        if ACTIVE_WORKFLOW["EMB_TYPE"]=="hugginface":
            self.embeddings=HuggingFaceEmbeddings(model_name=config.EMD_PATH)
        else:
            self.embeddings =  OpenAIEmbeddings(
                openai_api_base=config.api_base,
                openai_api_key=config.api_key,
                deployment=config.EMB_DEPLOYMENT_NAME,
                model=config.EMB_MODEL,
                openai_api_type='azure',)     
        
        
        
    
    def _init_db(self,data):
        persist_directory = ACTIVE_WORKFLOW["persist_directory"]
        docsearch = Chroma.from_texts(data, self.embeddings,persist_directory=persist_directory)
        docsearch.persist()
        return docsearch

    def _init_db_from_storage(self):
        persist_directory = ACTIVE_WORKFLOW["persist_directory"]
        docsearch=Chroma( embedding_function=self.embeddings,persist_directory=persist_directory)
        return docsearch
    
    def _preprocessor(self,data):
        if ACTIVE_WORKFLOW["textSpliter"]=="CTP":
            text_splitter = CharacterTextSplitter(
                                separator="\n",
                                chunk_size=ACTIVE_WORKFLOW["chunk_size"],
                                chunk_overlap=ACTIVE_WORKFLOW["chunk_overlap"],
                                length_function=len,
                            )
        elif ACTIVE_WORKFLOW["textSpliter"]=="RCTP":
        
            text_splitter = RecursiveCharacterTextSplitter(
                
                chunk_size=ACTIVE_WORKFLOW["chunk_size"],
                chunk_overlap=ACTIVE_WORKFLOW["chunk_overlap"],
                length_function=len,
            )
            
        
        if ACTIVE_WORKFLOW["splitOn"]=="document":
            chunks = text_splitter.split_documents(data)

        elif ACTIVE_WORKFLOW["splitOn"]=="text":
            text=""
            for p in data:
                text+=p.page_content
            chunks = text_splitter.split_text(text)
        
        return chunks
    def _prompt_template(self):
        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": PROMPT}
        return chain_type_kwargs
    

    def _qa(self,docsearch,chain_type_kwargs):
        '''
        self.qa_chain = RetrievalQA.from_chain_type(self.llm,
                                                    retriever=docsearch.as_retriever(search_type="mmr"),
                                                        return_source_documents=True,
                                                        chain_type_kwargs = chain_type_kwargs
                                                    )
        '''
        self.qa_chain=ConversationalRetrievalChain.from_llm(llm=self.llm,
                                                            retriever=docsearch.as_retriever(search_type="similarity",
                                                                                             search_kwargs={"k": 3}),
                                                            return_source_documents=True,
                                                            verbose=True,
                                                            combine_docs_chain_kwargs=chain_type_kwargs)

    def predict(self,query):
        global chat_history
        results=self.qa_chain({"question": query,"chat_history": chat_history[-3:]})
        
        info=[i.metadata for i in results["source_documents"]]
        output=results['answer']
        
        chat_history.append((query,output))
        
        return output,info




