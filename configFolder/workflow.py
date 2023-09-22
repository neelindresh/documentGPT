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
    "pdf":"contract_docs/KPMG_STL_Process Mining_LOE_v1.0 - 01.06.2022_Final.pdf",
    "folder":None,
    "persist_directory":"./storage_contract_v3",
    "chunk_size":1024,
    "chunk_overlap":500,
    "EMB_TYPE": "hugginface",
    "textSpliter":"CTP",
    "splitOn":"text",
    "model":"azurechatopenai"

}
SHELL_WORKFLOW={
    "pdf":"./shell_docs/shell-annual-report-2022.pdf",
    "folder":None,
    "persist_directory":"./storage_shell_v2_1024_500_full_CTP",
    "chunk_size":1024,
    "chunk_overlap":500,
    "EMB_TYPE": "hugginface",
    "textSpliter":"CTP",
    "splitOn":"text",
    "model":"azurechatopenai",

}
GSK_WORKFLOW={
    "pdf":"./GSK_docs/annual-report-2022.pdf",
    "folder":None,
    "persist_directory":"./storage_gsk_v2",
    "chunk_size":512,
    "chunk_overlap":200,
    "EMB_TYPE": "hugginface",
    "textSpliter":"CTP",
    "splitOn":"text",
    "model":"azurechatopenai"

}

Workflow=CONTRACT_WORKFLOW

