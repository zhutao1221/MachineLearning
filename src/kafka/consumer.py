# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 21:58:09 2019

@author: Administrator
"""
from kafka import KafkaConsumer
import datetime

consumer = KafkaConsumer('foobar',bootstrap_servers=['localhost:9092'])
for msg in consumer:
    #print(msg.key,msg.value)
    input_date = datetime.datetime.strptime(msg.value,'%Y-%m-%d %H:%M:%S.%f')
    now = datetime.datetime.now()
    print(now - input_date)
    
    
    #print("%s:%d:%d: key=%s value=%s" % (msg.topic, msg.partition,msg.offset, msg.key,msg.value))