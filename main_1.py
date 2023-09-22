from langchain.llms import AzureOpenAI
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings,OpenAIEmbeddings
from langchain.document_loaders import PyMuPDFLoader,PyPDFLoader
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA,ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFDirectoryLoader


import pandas as pd
import configFolder.config as config
llm = AzureOpenAI(
    openai_api_base=config.api_base,
    openai_api_version=config.api_version,
    deployment_name=config.deployment_name,
    openai_api_key=config.api_key,
    openai_api_type="azure",
    temperature=0.3
)


vectordb = Chroma( embedding_function=embedding,persist_directory=persist_directory)

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

chat_history=[]
#qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,)
qa= ConversationalRetrievalChain.from_llm(llm=llm,retriever=retriever, return_source_documents=True)
queries=["explain glass break precedure?","who is the author of the documents?","When was the document released?","can you explain access authorization in details?","who can provide it?",]
c=0
qa_ans=[]

while c<len(queries):
        user_input=queries[c]
        #user_input = input("Enter a query: ")
        if user_input == "exit":
            break
        
        query = f"###Prompt {user_input}"
        
        print(user_input)
        #llm_response = qa(user_input)
        llm_response=qa({"question": user_input, 'chat_history': chat_history[:-3]})
        print(llm_response)
        #print(ll)
        #results=post_processcor(llm_response["result"])
        results=post_processcor(llm_response["answer"])
        qa_ans.append({"query":user_input,"result":results})
        chat_history+=[(user_input,results)]
        
        c+=1

pd.DataFrame(qa_ans).to_csv("out.csv",index=False)
