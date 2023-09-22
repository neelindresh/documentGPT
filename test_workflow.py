from configFolder import config,workflow
from utils import model_loader,helper
import pandas as pd
import time
import sys

model=model_loader.MultiDocumentChatAzureOpenAI(verbose=True)

file_name=None
print(sys.argv)

with open("text_qa_con.txt","r") as f:
    qa=f.readlines()

save_ans=[]
print("Enter QA mode")
for q in qa:
    print(q)
    a=model.predict(q)
    print(a)
    save_ans.append({"question":q,"answer":a[0]})



pd.DataFrame(save_ans).to_csv("QA_CON.csv",index=False)