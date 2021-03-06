# -*- coding: utf-8 -*-
import cv2
import urllib
import numpy as np
from aip import AipFace
import base64
import multiprocessing
      
stream=urllib.urlopen('http://admin:admin@192.168.0.143:8081/video')
bytes=''

APP_ID = 'MyFaceApp'
API_KEY = 'YupM4GGtRLUxiiqHGwkeDGZY'
SECRET_KEY = 'IcZ8MqZkGwZTp7XY5u1ZIXpMZBniw2Nx'
imageType = 'BASE64'
groupIdList = "hz"

tStr = 'error_code'
errorCodeKey = tStr.decode('unicode-escape')
tStr = 'result'
resultKey = tStr.decode('unicode-escape')
tStr = 'user_list'
userListKey = tStr.decode('unicode-escape')
tStr = 'user_id'
userIdKey = tStr.decode('unicode-escape')
 
client = AipFace(APP_ID, API_KEY, SECRET_KEY)

def execClient(ijpg):
    result = client.search(ijpg, imageType, groupIdList)
    if result[errorCodeKey] == 0:
        print(result[resultKey][userListKey][0])
        print(result[resultKey][userListKey][0][userIdKey])

index = 0
while True:
    index = index + 1
    bytes+=stream.read(1024*8)
    a = bytes.find('\xff\xd8')
    b = bytes.find('\xff\xd9')
    
    if a!=-1 and b!=-1:
        jpg = bytes[a:b+2]
        jpg4AipFace = base64.b64encode(jpg)        
        bytes= bytes[b+2:]
        ipcam = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        cv2.imshow('ipcam',ipcam)        
        if cv2.waitKey(1) ==27:
            exit(0)
        if index%25 == 0:
            t = multiprocessing.Process(target=execClient,args=(jpg4AipFace,))
            t.daemon=True#将daemon设置为True，则主线程不比等待子进程，主线程结束则所有结束
            t.start()
            cv2.imwrite('temp/imageAI/image.jpg',ipcam) #存储为图像