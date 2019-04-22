import timeit
import time
import ray
import numpy as np
from multiprocessing import Pool

a = 0
b = 0
c = 0
d = 0
e = 0
f = 0
grande = np.array([0,0,0,0,0,0])

def normal():
	global a
	global b
	global c
	global d
	global e
	global f
	
	a = a+1
	b = b+1
	c = c+1
	d = d+1
	e = e+1
	f = f+1

def lista():
	global grande
	grande = np.add(grande,1)

def soma1(numero):
	return numero+1

@ray.remote
def parRay():
	global a
	global b
	global c
	global d
	global e
	global f
	
	a = a+1
	b = b+1
	c = c+1
	d = d+1
	e = e+1
	f = f+1

#ray.init()
pool = Pool()
print("=======================")
t1 = time.perf_counter()
for i in range(400*400):
	normal()
t2 = time.perf_counter()
print("Normal: ",t2-t1)

t1 = time.perf_counter()
for i in range(400*400):
	lista()
t2 = time.perf_counter()
print("Lista: ",t2-t1)

t1 = time.perf_counter()
for i in range(400*400):
	args =[a,b,c,d,e,f]
	a,b,c,d,e,f = pool.map(soma1,args)
t2 = time.perf_counter()
print("Multiproc: ",t2-t1)




import code
code.interact(local=locals())