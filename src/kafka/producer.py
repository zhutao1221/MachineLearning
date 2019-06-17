# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 21:53:24 2019

@author: Administrator
"""
from kafka import KafkaProducer
import datetime
import time

producer = KafkaProducer(bootstrap_servers='localhost:9092')
for _ in range(20):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    producer.send('foobar', key=b'foo', value=now)
    time.sleep(0.001)