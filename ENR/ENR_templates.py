from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
from dataclasses import dataclass,asdict
from langchain_community.chat_models import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
import pandas as pd


VECTOR_DBS={
    "tender":"ENR_TENDERS",
    "regulation":"ENR_REGULATION",
    "summary":"ENR_TENDER_SUMMARY"
}

classification_prompt='''
Given a user query, Classify it into Either `Regulation` or `Tender` 

Return `Regulation` if the query is related to the below table 

```Regulation Details

{reg_mapping}
```

Return `Tender` if the query is in the following context

{tender_summary}


// Regulation: If the user query is related to regulations
// Tender: If it is related to the Tender documents or a general question where user is asking applied regulations


User Query
---------
{query}
'''

regclassificationChatTemplate=PromptTemplate(
    input_variables=["query","reg_mapping","tender_summary"],
    template=classification_prompt,
)



tender_chat_prompt='''
You are a helpful assitant who is master in tender analysis.

Given the content, please provide the answer to the user queryas good as possible

Previous Chat History
---------
{chat_history}

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
    input_variables=["query","context","summary","chat_history"],
    template=tender_chat_prompt,
)


regulation_prompt='''
You are a helpful assitant who is master in tender analysis.

Given the content, please provide the answer to the user queryas good as possible

Previous Chat History
---------
{chat_history}

Context
---------
{context}

User Query
---------
{query}
'''

regulationChatTemplate=PromptTemplate(
    input_variables=["query","context","chat_history"],
    template=regulation_prompt,
)