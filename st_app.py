# Import necessary libraries
import streamlit as st
from streamlit_chat import message
import tempfile
#from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from configFolder import config
from langchain.llms import AzureOpenAI
from utils import model_loader,helper
# Define the path for generated embeddings


st.title("PDF GPT")
model=model_loader.MultiDocumentChatAzureOpenAI()
#model=model_loader.ConversationMultiDocumentAzureOpenAI()
#model=model_loader.ConversationAzureOpenAI()
# Initialize chat history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Initialize messages
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me  🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]

# Create containers for chat history and user input
response_container = st.container()
container = st.container()

# User input form
with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Talk to PDF data 👉 (:", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output,info = model.predict(user_input)
        print(info)
        print(output)
        output=helper.post_processcor(output)
        output=output.replace('"""',"")
        output=output.replace('# text =',"")
        output=output.replace('# ',"")
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

# Display chat history
if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', )
            message(st.session_state["generated"][i], key=str(i),)