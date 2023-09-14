#EMB_TYPE:  hugginface -> local embedding or opensouce Embeddings | OpenAI
#textSpliter-> CTP: CharacterTextSplitter | RCTP: RecursiveCharacterTextSplitter
# splitOn-> document: DocumentLevel Splitting with metadata | text: TextLevel Splitting no metadata
# model->azureopenai: AzureOpenAI services | azurechatopenai: AzureOpenAIChat Services | hugginfaceT5model: Any opensource T5 Model version

HP_WORKFLOW={
    "pdf":None,
    "folder":"./hp_docs/",
    "persist_directory":"./storage_hp",
    "chunk_size":512,
    "chunk_overlap":200,
    "EMB_TYPE": "hugginface",
    "textSpliter":"RCTP",
    "splitOn":"document",
    "model":"azureopenai"

}

TATA_STEEL_WORKFLOW={
    "pdf":None,
    "folder":"./tata_steels_docs",
    "persist_directory":"./storage_tata",
    "chunk_size":1024,
    "chunk_overlap":300,
    "EMB_TYPE": "hugginface",
    "textSpliter":"CTP",
    "splitOn":"text",
    "model":"azurechatopenai"
}

CONTRACT_WORKFLOW={
    "pdf":None,
    "folder":"./contract_docs",
    "persist_directory":"./storage_contract",
    "chunk_size":512,
    "chunk_overlap":200,
    "EMB_TYPE": "hugginface",
    "textSpliter":"CTP",
    "splitOn":"text",
    "model":"azurechatopenai"

}

Workflow=CONTRACT_WORKFLOW

