
import requests

#url = 'http://20.41.249.147:6061/uploadfile/TEST123'
url = 'https://ikegai.southindia.cloudapp.azure.com/agent/uploadfile/MULTI123'

#url='http://127.0.0.1:6069/agent/uploadfile/TEST123'
#file = {'file': open('Profile.pdf', 'rb')}
file=[('file',open('table.pdf', 'rb'))]
resp = requests.post(url=url, files=file) 
print(resp.json())

'''
import requests
import json
import time
url = 'http://127.0.0.1:8000/getresponce/'

for i in ["who is indresh",'what is his full name',"what are his skills","what is his email id","what work did he do in  PRM Fincon"]:
    data={'query':i}
    resp = requests.post(url=url,json=data) 
    print(resp.json())
    time.sleep(10)

uid="TEST123"

res=requests.get(f"https://ikegai.southindia.cloudapp.azure.com/solution-manager/v1/useCase/usecase-by-id?id={uid}")
print(res.status_code)
'''