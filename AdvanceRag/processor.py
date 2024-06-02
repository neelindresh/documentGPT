import langchain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import AzureChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain_core.documents.base import Document
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import os
from dataclasses import dataclass,asdict
import fitz
import json
from langchain_community.document_loaders import PyMuPDFLoader
from pydantic import BaseModel, Field
from typing import List
import os
import pandas as pd
from IPython.display import Markdown as md
import fitz
from tqdm import tqdm
import chromadb
from chromadb.config import Settings

from AdvanceRag.prompt_utils import get_meta_extractor
from AdvanceRag.schema import MetaDataOfDocuments
import config

class Embeddings:
    def __init__(self,name) -> None:
        self.name=name
    def load(self):
        return HuggingFaceEmbeddings(model_name=self.name)

    
class DocumentProcessor:
    def __init__(self,pdf_path,end_point,api_key):
        self.document_analysis_client = DocumentAnalysisClient(
            endpoint=end_point, credential=AzureKeyCredential(api_key)
        )
        self.pdf_path=pdf_path
        self.docs=fitz.open(self.pdf_path)
        self.page_info={}
        self.block_info=[]
        
    def process(self,page_range:str=""):
       
        with open(self.pdf_path, "rb") as f:
            if page_range!="":
                poller = self.document_analysis_client.begin_analyze_document(
                    "prebuilt-layout", document=f,pages=page_range)
            else:
                poller = self.document_analysis_client.begin_analyze_document(
                    "prebuilt-layout", document=f)
        result=poller.result()
        tables=self.table_extraction(result)
        #print(tables)
        page_info=self.make_markdown(result,tables)
            
        return self.page_info
    def fill_cols(self,rows):
        max_len=max([len(v) for k,v in rows.items()])
        for row in rows:
            if len(rows[row])<max_len:
                for i in range(max_len):
                    if i not in rows[row]:
                        rows[row][i]=""
                rows[row]=dict(sorted(rows[row].items()))
        return rows
    def table_extraction(self,result):
        table_info=[]
        for idx,tab in enumerate(result.tables):
            table_bb_box=tab.bounding_regions[0]
            page_no=table_bb_box.page_number
            rows={}
            for cell in tab.cells:
                #print(cell.row_index,cell.column_index,cell.content)
                if cell.content!="":
                    if cell.row_index in rows:
                        rows[cell.row_index][cell.column_index]=cell.content
                    else:
                        rows[cell.row_index]={
                            cell.column_index:cell.content
                        }
            rows=self.fill_cols(rows)
            #print(rows)
            df=pd.DataFrame(rows).T
            #print(df)
            table_info.append({
                "table":df,
                "bbox":table_bb_box,
                "table_name":f"table_{page_no}_{idx}",
                "page_number":page_no
            })
        return table_info
    
    def check_if_in_tables(self,tables,para_data,page_no):
        page_no=para_data["page_no"]
        para_coor=para_data['bbox']
        #print("==========PARA_COOR=========")
        (x1,y1),(x2,y2)=para_coor[0],para_coor[1]
        #print((x1,y1),(x2,y2))
        flag=False
        table_name=""
        for t in tables:
            #print("==========TABLE_COORDINATES=========")
            if t["page_number"]==page_no:
                bbox=t['bbox'].polygon
                (tab_x1,tab_y1),(tab_x2,tab_y2)=(bbox[0].x,bbox[0].y),(bbox[2].x,bbox[2].y)
                #print((tab_x1,tab_y1),(tab_x2,tab_y2))
                #print((y1>tab_y2 or y2<tab_y1),(x1>tab_x2 or x2<tab_x1),(y1>tab_y2 or y2<tab_y1) or (x1>tab_x2 or x2<tab_x1))
                if (y1>tab_y2 or y2<tab_y1) or (x1>tab_x2 or x2<tab_x1):
                    pass
                else:
                    return True, t['table_name'],t['table'].fillna("").to_markdown(),{"x1":tab_x1,"y1":tab_y1,"x2":tab_x2,"y2":tab_y2}
        return flag,_,_,_
    
    def make_markdown(self,result,table_data):
        info={}
        last_title=""
        escape_tags=["pageFooter","pageNumber"]
        
        last_table_merged=""
        for para in result.paragraphs:
            #print(para.content)
            page_no=para.bounding_regions[0].page_number
            #print("page_no:",page_no)
            bbox=para.bounding_regions[0].polygon

            (x1,y1),(x2,y2)=(bbox[0].x,bbox[0].y),(bbox[2].x,bbox[2].y)
            para_data={"bbox":[(x1,y1),(x2,y2)],"page_no":page_no}
            table_flag,table_name,table,table_bbox=self.check_if_in_tables(table_data,para_data,page_no)
            #print(table_flag)
            #is on (top or bottom)
            if page_no not in self.page_info:
                self.page_info[page_no]=""
            if not table_flag:
                #print(para.content)
                if para.role=='title':
                    #info[para.content]=[]
                    #last_title=para.content
                    self.page_info[page_no]+=f"# {para.content}  \n  \n"
                    self.block_info.append({
                        "content": para.content,
                        "bbox":{
                            "x1":x1,
                            "y1":y1,
                            "x2":x2,
                            "y2":y2
                        },
                        "page_no":page_no,
                        "role":"header"
                    })
                elif para.role in escape_tags:
                    continue
                else:
                    #info[last_title].append(para.content)
                    self.page_info[page_no]+=f"{para.content}  \n  \n"
                    self.block_info.append({
                        "content": para.content,
                        "bbox":{
                            "x1":x1,
                            "y1":y1,
                            "x2":x2,
                            "y2":y2
                        },
                        "page_no":page_no,
                        "role":"paragraph"
                    })
            else:
                
                if table_name!=last_table_merged:
                    last_table_merged=table_name

                    self.page_info[page_no]+=f"{table}  \n  \n"
                    
                    self.block_info.append({
                        "content": table,
                        "bbox":table_bbox,
                        "page_no":page_no,
                        "role":"table"
                    })
        return self.page_info
    
    
class MetaDataRichDocumentProcessor:
    def __init__(self,azure_doc_creds,azure_openai_creds,chroma_creds,embeddings):
        self.azure_doc_creds=azure_doc_creds
        self.llm=AzureChatOpenAI(**azure_openai_creds)
        self.meta_data_extraction_chain=get_meta_extractor(llm=self.llm)
        self.embeddings=Embeddings(embeddings).load()
        self.client= chromadb.HttpClient(**chroma_creds)
    def process(self,doc_paths:list,collection_name,page_range=None):
        """_summary_

        Args:
            doc_path (_type_): _description_
            page_range (_type_, optional): page range in format of '1-5'. Defaults to None.
        """
        metastorage=[]
        data_processed={}
        coll_name=[]
        for idx,j in enumerate(doc_paths):
            if j.endswith(".pdf"):
                pg_data=DocumentProcessor(j,end_point=self.azure_doc_creds.end_point,api_key=self.azure_doc_creds.api_key)
                processed_doc=pg_data.process()
                out=self.meta_data_extraction_chain.invoke({
                    "context": pg_data.page_info[1],
                    "schema": MetaDataOfDocuments.model_json_schema()
                })
                meta_store=json.loads(out.content)
                meta_store["filename"]=j
                
                data_processed[j]=pg_data.page_info
                docs=[]
                for d in pg_data.page_info:
                    md={"page":d,"path":j}
                    md.update({k:v for k,v in meta_store.items() if k.lower()!="topics"})
                    doc=Document(page_content=pg_data.page_info[d],metadata=md)
                    docs.append(doc)
                vdb=Chroma(embedding_function=self.embeddings,persist_directory="TestRagv1",client=self.client)
                vdbv=vdb.from_documents(docs,embedding=self.embeddings,persist_directory="TestRagv1",collection_name=f'{collection_name}_{idx}',client=self.client)
                vdbv.persist()
                meta_store["collection_name"]=f'{collection_name}_{idx}'
                coll_name.append(f'{collection_name}_{idx}')
                metastorage.append(meta_store)
        ## CODE -- Save MetaStore to Usecase Level Details -mostly in `data_source``