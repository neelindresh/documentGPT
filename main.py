from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse

from fastapi.middleware.cors import CORSMiddleware

import os


from utils.model_utils import ConvertToVector,LLMmodel,LLMmodelV1,AzureDocIntell,CompartiveAnalysis
from pydantic import BaseModel

from typing import Optional, Type, Any, Tuple
from dataclasses import asdict

from config import AzureDocumentInfo

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

file_upload_path=os.path.join("TBD","file_uploads")
vectordb_store_path=os.path.join("TBD","vectordb")
emd_name="intfloat/e5-base-v2"
azure_form= AzureDocIntell(**asdict(AzureDocumentInfo()))
vectorizer=ConvertToVector("intfloat/e5-base-v2",azure_form)
model=LLMmodelV1(embeddings=emd_name,db_name=os.path.join(vectordb_store_path,'DUMMY'))

comarative_analysis=CompartiveAnalysis(embeddings=emd_name,db_name="CompetetorRagV1")
@app.post("/uploadfile/{idx}")
async def create_upload_file(file: UploadFile,idx:str):
    
    contents = file.file.read()
    with open(os.path.join(file_upload_path,file.filename), 'wb') as f:
        f.write(contents)
    vectorizer.convert_to_vector(os.path.join(file_upload_path,file.filename),vectordb_store_path,idx)
    
    return {"filename": file.filename}


@app.post('/getresponce/')
async def get_responce(query:dict):
    if 'params' in query:
        model._set_llm(query['params'])
    if 'filename' in query:
        if query['filename']:
            model._set_vdb(os.path.join(vectordb_store_path,query['filename']))
    out=model.predict(query['query'])
    return out


@app.get("/all_databases/")
async def all_databases():
    return [i for i in os.listdir(vectordb_store_path)]


@app.get("/download/{filename}")
async def dowload(filename:str):
    pdf_name_mapping={'az1742-2018.pdf':'Solar Photovoltic (PV) System Components.pdf',
                          '6981.pdf':"Photovoltics: Basic Design Princicals and Components.pdf",
                          'BOOK3.pdf':"Solar Photovoltics Technology and Systems.pdf"
                          }
    reverse_mapping={v:k for k,v in pdf_name_mapping.items()}
    if filename in reverse_mapping:
        filename=reverse_mapping[filename]
    file_path=os.path.join(file_upload_path,filename)
    return FileResponse(path=file_path, filename=file_path, media_type='text/pdf')


@app.post('/getcomparative/')
async def getcomparative(query:dict):
    
    out=comarative_analysis.predict(query['query'])
    return out