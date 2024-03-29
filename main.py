from fastapi import FastAPI, UploadFile
import os


from utils.model_utils import ConvertToVector,LLMmodel,LLMmodelV1
from pydantic import BaseModel

from typing import Optional, Type, Any, Tuple


app = FastAPI()
file_upload_path=os.path.join("TBD","file_uploads")
vectordb_store_path=os.path.join("TBD","vectordb")
emd_name="intfloat/e5-base-v2"
vectorizer=ConvertToVector("intfloat/e5-base-v2")
model=LLMmodelV1(embeddings=emd_name,db_name=os.path.join(vectordb_store_path,'DUMMY'))
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    contents = file.file.read()
    with open(os.path.join(file_upload_path,file.filename), 'wb') as f:
        f.write(contents)
    vectorizer.convert_to_vector(os.path.join(file_upload_path,file.filename),os.path.join(vectordb_store_path,file.filename))
    
    return {"filename": file.filename}


@app.post('/getresponce/')
async def get_responce(query:dict):
    if 'filename' in query:
        if query['filename']:
            model._set_vdb(os.path.join(vectordb_store_path,query['filename']))
    out=model.predict(query['query'])
    return out


@app.get("/all_databases/")
async def all_databases():
    return [i for i in os.listdir(vectordb_store_path)]