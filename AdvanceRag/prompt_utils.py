from AdvanceRag.prompt_templates import meta_data_extraction_template,meta_data_filtration_template,rag_template, reranking_template
from langchain.prompts import PromptTemplate

def get_meta_extractor(llm):
    prompt=PromptTemplate.from_template(meta_data_extraction_template)
    chain=prompt |llm 
    return chain

def get_meta_filtration(llm):
    prompt=PromptTemplate.from_template(meta_data_filtration_template)
    chain=prompt |llm 
    return chain

def get_reranker(llm):
    prompt=PromptTemplate.from_template(reranking_template)
    chain=prompt | llm 
    return chain

def get_rag(llm):
    prompt=PromptTemplate.from_template(rag_template)
    chain=prompt | llm 
    return chain
    