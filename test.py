
import requests

#url = 'http://20.41.249.147:6061/uploadfile/TEST123'
url = 'https://ikegai.southindia.cloudapp.azure.com/agent/uploadfile/TEST123'

#url='http://127.0.0.1:6069/agent/uploadfile/TEST123'
#file = {'file': open('Profile.pdf', 'rb')}
file=[('file',open('SECI000126-3219607-RfSfor1000MW-FDRE-V-finalupload.pdf', 'rb')),('file',open('175-Notification.pdf', 'rb'))]
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