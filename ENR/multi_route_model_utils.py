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
from IPython.display import Markdown as md
import fitz
from tqdm import tqdm


import chromadb
from chromadb.config import Settings

from ENR.multi_route_model_chains import router_chain,metafinder,regulation_chat,tender_chat, tender_regulation_miner,get_agent_brain

    





    






