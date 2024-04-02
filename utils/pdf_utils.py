import fitz 
import pandas as pd
from langchain_core.documents.base import Document
import os
import shutil
def convert_to_langchain_docs(docs):
    document_format=[]
    for doc in docs:
        document_format.append(Document(page_content=doc['block'],metadata={"page":doc['page_no'],"path":doc['doc_name']}))
    return document_format

def process(pdf_path:str):
    doc=fitz.open(pdf_path)
    return extract_doc(doc, pdf_path)

def get_data(docs,page:int=None,page_range:list=None):
    if page:
        page_docs=[]
        for doc in docs:
            if doc['page_no']==page:
                page_docs.append(doc)
        return page_docs
    if page_range:
        page_list=list(range(page_range[0],page_range[1]))
        page_docs=[]
        for doc in docs:
            if doc['page_no'] in page_list:
                page_docs.append(doc)
        return page_docs
    
    return docs
 
def extract_doc(docs,doc_name:str,clean:bool=True):
    data_format:list=[]
    clean_list:list=["\n"]
    for page_no in range(docs.page_count):
        para:list=[]
        last_para:str=""
        starts_with:tuple=('',"(","•","¾")
        merge_next:bool=False
        merge_next_char:str=""
        for b in docs[page_no].get_text("blocks"):
            x0,y0,x1,y1,text,block_id,block_type=b
            if clean:
                for c in clean_list:
                    text=text.replace(c,"")
                    text=text.strip()
            if text.strip()=="":
                continue
            if merge_next:
                if len(para)==0:
                    para.append("")
                para[-1]=last_para+text
                last_para+=text
                op,character=merge_next_char.split("_")
                if op=="HAS":
                    if character in text:
                        merge_next=False
                        merge_next_char=""
                elif op=="ENDS":
                    if text.strip().endswith(character):
                        merge_next=False
                        merge_next_char=""
                continue

            if "(" in text and ")" not in text:
                merge_next_char="HAS_)"
                merge_next=True
            elif "[" in text and "]" not in text:
                merge_next_char="HAS_]"
                merge_next=True
            elif "{" in text and "}" not in text:
                merge_next_char="HAS_}"
                merge_next=True
            if text.startswith(starts_with):
                if len(para)==0:
                    para.append("")
                para[-1]=last_para+text
                last_para+=text
                continue
            if not text.strip().endswith("."):
                merge_next=True
                merge_next_char="ENDS_."
            elif text.strip()[0].islower():
                
                if len(para)==0:
                    para.append("")
                para[-1]=last_para+text
                last_para+=text
                continue

            para.append(text)
            last_para=text
            
        for p in para:
            data_format.append({
                "doc_name":doc_name,
                "page_no":page_no,
                "block":p
            })
    return data_format



def check_if_scanned(page: str,threshold:float=0.05) -> tuple:
    """
    Calculate the percentage of document that is covered by (searchable) text.

    If the returned percentage of text is very low, the document is
    most likely a scanned PDF
    """
    total_page_area = 0.0
    total_text_area = 0.0

    
    total_page_area = total_page_area + abs(page.rect)
    text_area = 0.0
    for b in page.get_text_blocks():
        x0,y0,x1,y1,text,block_id,block_type=b
        if not text.startswith("<image"):
            r = fitz.Rect(b[x0,y0,x1,y1])  # rectangle where block text appears
            text_area = text_area + abs(r)
    total_text_area = total_text_area + text_area
    text_perc=total_text_area / total_page_area
    return text_perc < threshold, text_perc
def check_if_scanned_full_doc(path: str,threshold=0.05) -> float:
    """
    Calculate the percentage of document that is covered by (searchable) text.

    If the returned percentage of text is very low, the document is
    most likely a scanned PDF
    """
    total_page_area = 0.0
    total_text_area = 0.0
    
    doc = fitz.open(path)

    for page_num, page in enumerate(doc):
        total_page_area = total_page_area + abs(page.rect)
        text_area = 0.0
        for b in page.get_text_blocks():
            x0,y0,x1,y1,text,block_id,block_type=b
            if not text.startswith("<image"):
                r = fitz.Rect(b[:4])  # rectangle where block text appears
                text_area = text_area + abs(r)
        total_text_area = total_text_area + text_area
    doc.close()
    text_perc=total_text_area / total_page_area
    return text_perc < threshold, text_perc

def convert_to_doc_intell_pdf_format(pdf_file:str,pages=None,out_dir="temp"):
    doc1 = fitz.open(pdf_file)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    else:
        shutil.rmtree(out_dir)
        os.mkdir(out_dir)
    # Create a new empty PDF document
    filenames=[]
    if pages is None:
        # Insert the first 2 pages of doc1 into doc2
        for p in range(len(doc1)):
            doc2 = fitz.open()
            doc2.insert_pdf(doc1, to_page=p+1,from_page=p)
            
            # Save the modified document as "first-and-last-10.pdf"
            fname=f"{out_dir}/_temp_pdf_{p}.pdf"
            doc2.save(fname)
            filenames.append(fname)
            
            
    ## Implement if pages are given
    
    return filenames