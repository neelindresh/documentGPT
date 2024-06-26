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
from typing import List,Literal
import os
import pandas as pd
import fitz
from tqdm import tqdm


import chromadb
from chromadb.config import Settings

def router_chain(llm):

    prompt='''
    Given a user query, Classify it into Either `Regulation`, `Tender`, `Summary`, `OpenTender`

    // Regulation: If the user query is related to regulations
    // Tender: If it is related to the Tender documents or a general question where user is asking applied regulations
    // Summary: If the query is about a summarizing the document, or its content
    // OpenTender: If the Query is related to the open tenders


    Examples:
    Input: What are the details available in the bid information sheet?
    Output: Tender

    Input: What is the brief scope of work of the RfS?
    Output: Tender

    Input: What is the overall background of the RfS?
    Output: Tender

    Input: What are the eligibility criteria for the project?
    Output: Tender

    Input: What can be done in case excess power is generated by the project?
    Output: Tender

    Input: What are the application costs and procedures for grant of connectivity?
    Output: Regulation
    
    Input: Provide an overview of the grid code applicable for the project
    Output: Regulation
    
    Input: Can additional General Network Access be granted
    Output: Regulation

    Input: What are the grid connectivity standards and regulations for new generating units?
    Output: Regulation

    Input: Provide a synopsis of the General Network Access regulations governing the project
    Output: Regulation

    Input: Who is eligible for general network access and/or connection to the ISTS?
    Output: Regulation

    Input: Can you provide list of open Tenders
    Output: OpenTender

    Input: Can you give me a section wise summary of Tender 0
    Output: Summary
    Input: Can you summarize the Sections of the rfs
    Output: Summary

    Input: Can you give the list of sections in the document?
    Output: Summary

    Input: Can you give the list of sections in the document?
    Output: Summary

    Input: What regulations does the tender require the bidder to be compliant with?
    Output: Tender,Regulation

    Input: What is the distribution of regulation references across various sections of the document?
    Output: Tender,Regulation

    Input: What are the regulatory bodies governing various activities under the RfS?
    Output: Tender,Regulation

    Input: {query}
    Output: 
    '''

    routerChatTemplate=PromptTemplate(
        input_variables=["query"],
        template=prompt,
    )
    reg_class_chain=routerChatTemplate | llm
    return reg_class_chain

def tender_chat(llm):
    tender_chat_prompt='''You are a helpful assitant who is master in tender analysis.

    Given the content, please provide the answer to the user query as good as possible. Please provide as much details as possible

    Summarized information
    ---------
    {summary}

    Context
    ---------
    {context}

    User Query
    ---------
    {query}
    '''

    chatTemplate=PromptTemplate(
        input_variables=["query","context","summary"],
        template=tender_chat_prompt,
    )
    return chatTemplate | llm

def regulation_chat(llm):
    regulation_prompt='''
    You are a helpful assitant who is master in tender analysis.

    Given the content, please provide the answer to the user query as good as possible

    Context
    ---------
    {context}

    User Query
    ---------
    {query}
    '''

    regulationChatTemplate=PromptTemplate(
        input_variables=["query","context"],
        template=regulation_prompt,
    )
    return regulationChatTemplate | llm

def metafinder(llm):
    template = """You are Database Admin, given the following MetaStore return all the `collection_name`  which might have some information on the about the `user_query`
    Output should follow the pattern defined in output Format.
    No verbose should be present. Output should follow the pattern defined in schema
    
    MetaStore
    -----------
    {metastore}
    
    output Format
    -------------
    [collection1, collection2]
    
    user_query:{query}
    //Note: Don't change the output Format, dont add anything else into it
    """
    meta_finder_template = PromptTemplate.from_template(template)
    meta_finder = meta_finder_template | llm
    return meta_finder

def tender_regulation_miner(llm):
    template = """You are a helpful AI assitant. Given the content of the Tender sumary and Regulation summary, please provide the answer to the user query as good as possible
    
    Regulation
    -----------
    {regulation}
    
    Tender
    -------------
    {tender}
    
    user_query:{query}
    """
    meta_finder_template = PromptTemplate.from_template(template)
    meta_finder = meta_finder_template | llm
    return meta_finder






    
def get_agent_brain(llm):
    #prev_history=convert_to_string(chat_history)
    regulation_prompt='''
    You are a helpful assitant, Please help in providing correct information.
    
    Given the Previous Conversation, Do you have enough information to give answer.
    
    If `Yes` Please return the valid responce if not please say `No`, If `No` do not add any verbose.
    
    // If you dont have an answer please refrain from answering just say `No`
    // You are limited to the context of your Previous Chat history.

    Previous Conversation History
    ---------
    {chat_history}

    User Query
    ---------
    {query}
    '''

    regulationChatTemplate=PromptTemplate(
        input_variables=["query","context"],
        template=regulation_prompt,
    )
    return regulationChatTemplate | llm
    
