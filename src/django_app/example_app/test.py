# -*- coding: utf-8 -*-
import requests
import json
idata = json.dumps({'text': '你好'})
iheaders = {'Accept':'*/*',
           'Accept-Encoding':'gzip, deflate, br',
           'Accept-Language':'zh-CN,zh;q=0.8',
           'Connection':'keep-alive',
           'Content-Length':'13',
           'Content-Type':'application/json',
           'Cookie':'__guid=96992031.2623246281634526700.1509243307748.2983; sessionid=3f44sdfi7gxca42t0r889j2dtbpv6ip2; monitor_count=3',
           'Host':'127.0.0.1:8000',
           'Origin':'http://127.0.0.1:8000',
           'Referer':'http://127.0.0.1:8000/',
           'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
           'X-CSRFToken':'undefined',
           'X-Requested-With':'XMLHttpRequest'          
           }
r = requests.post('http://127.0.0.1:8000/api/chatterbot/', data=idata, headers=iheaders) 
print(r.text)