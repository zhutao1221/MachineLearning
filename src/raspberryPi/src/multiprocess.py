# -*- coding: utf-8 -*-
import multiprocessing
import time
 
def f0(a1):
    time.sleep(3)
    print(a1)

 
t = multiprocessing.Process(target=f0,args=(12,))
t.daemon=True#将daemon设置为True，则主线程不比等待子进程，主线程结束则所有结束
t.start()
 
t2 = multiprocessing.Process(target=f0, args=(13,))
t2.daemon = True
t2.start()
 
print('end')#默认情况下等待所有子进程结束，主进程才结束