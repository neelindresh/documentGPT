from langchain.llms import AzureOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


import configFolder.config as config
llm = AzureOpenAI(
    openai_api_base=config.api_base,
    openai_api_version=config.api_version,
    deployment_name=config.deployment_name,
    openai_api_key=config.api_key,
    openai_api_type="azure",
    temperature=0.5
)


# Run the LLM
#print(llm("Tell me a joke"))


loader = PyMuPDFLoader("data/Access-Control-Standards_Latest.pdf")

data = loader.load()
print(data[0])

print("Loading Embeddings")
embedding=HuggingFaceEmbeddings(model_name=config.EMD_PATH)
print("Embeddings Loaded")

text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=300,
        length_function=len,
    )
text=""
for p in data:
    text+=p.page_content
chunks = text_splitter.split_documents(data)

docsearch = Chroma.from_documents(data, embedding)

#knowledgeBase = FAISS.from_documents(chunks, embedding)


#chain = load_qa_chain(llm, chain_type='stuff')
query="give me the clauses for Provision/De-Provisioning?"
#docs = knowledgeBase.similarity_search(query)
#print(docs)
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
#response = chain.run(input_documents=knowledgeBase, question=query)
#print(response)
chain_type_kwargs = {"prompt": PROMPT}

qa_chain = RetrievalQA.from_chain_type(llm,retriever=docsearch.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': 0.8}), return_source_documents=True,chain_type_kwargs = chain_type_kwargs)

results=qa_chain({"query": query})
print(results['result'])
print([i.metadata['page'] for i in results['source_documents']])
