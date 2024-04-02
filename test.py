
import requests

url = 'http://127.0.0.1:8000/uploadfile/ResumeParser'
file = {'file': open('Karnataka Renewable.pdf', 'rb')}
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
'''