
import requests

url = 'http://20.41.249.147:6061/uploadfile/TEST123'
#url = 'http://127.0.0.1:8000/uploadfile/COMP123'
file = {'file': open('TATA STEELS 3qfy24-transcript-v7.pdf', 'rb')}
resp = requests.post(url=url, files=file,json={"vector":False}) 
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
'''