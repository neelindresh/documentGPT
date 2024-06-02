from pymongo import MongoClient
from dataclasses import dataclass,asdict
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
import json
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import tiktoken
from langchain_core.documents import Document
import tqdm
import numpy as np

from langchain_community.embeddings import HuggingFaceEmbeddings
from FlagEmbedding import FlagReranker
from config import ChromaClient
MAX_TOKEN_CONTEXT_LIMIT=7000

EMBEDDING="mixedbread-ai/mxbai-embed-large-v1"
RERANKER_MODEL='BAAI/bge-reranker-base'

def load_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name=EMBEDDING)
    return embeddings


def load_reranker():
    return FlagReranker(RERANKER_MODEL) # Setting use_fp16 to True speeds up computation with a slight performance degradation

def get_document_source(docs):
    info_list=[d.metadata for d in docs]
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

def make_documents_from_chroma(document_chunks):
    return [Document(page_content=d,metadata=m or {}) for d,m in zip(document_chunks['documents'],document_chunks['metadatas'])]

def ensemble_retriver(query,collections,client,embeddings):
    documents=[]
    for collection in tqdm.tqdm(collections):
        collection_name=collection['collection_name']
        vdb=Chroma(client=client,collection_name=collection_name,embedding_function=embeddings)
        document_chunks=vdb.get()
        bm_docs=make_documents_from_chroma(document_chunks)
        bm25_retriver=BM25Retriever.from_documents(bm_docs)
        retriver=vdb.as_retriever()
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriver, retriver], weights=[0.5, 0.5]
        )
        retrived_docs=ensemble_retriever.invoke(query)
        doc_list=[]
        for d in retrived_docs:
            d.page_content+="## Metadata:"+json.dumps(d.metadata)
            doc_list.append(d)
        documents.extend(doc_list)
    return documents

def rerank_docs(query,documents,reranker):
    fscores_n=[]
    for d in tqdm.tqdm(documents):
        #print(d.page_content)
        score = reranker.compute_score([query, d.page_content])
        fscores_n.append(score)
    return fscores_n

def get_final_documents(sorted_idx,scores,documents,max_token_limit):
    encoding = tiktoken.get_encoding("cl100k_base")
    final_docs=[]
    context_limit=0
    for d_index in sorted_idx:
        doc=documents[d_index]
        if scores[d_index]>0 and context_limit<max_token_limit:
            encoded=encoding.encode(doc.page_content)
            context_limit+=len(encoded)
            final_docs.append(doc)
        else:
            break
    return final_docs
        
