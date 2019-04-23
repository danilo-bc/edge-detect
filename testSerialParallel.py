import time
import numpy as np
import numpy.linalg as lin
import ray
import itertools

ray.init()

sizeness = 4*20
matriz = np.random.randint(5, size=(sizeness,sizeness))
print(matriz)
def div4(mat,tam):
	divided = []
	for i in range(0,tam-1,int(tam/2)):
		for j in range(0,tam-1,int(tam/2)):
			subm = matriz[np.ix_(range(i,i+int(tam/2)),range(j,j+int(tam/2)))]
			divided.append(subm)
	return divided

print("=======================")
t1 = time.time()
dets_serial = []
for i in range(0,sizeness-1,2):
	for j in range(0,sizeness-1,2):
		subm = matriz[np.ix_(range(i,i+2),range(j,j+2))]
		dets_serial.append(lin.det(subm))
print(len(dets_serial))
t2 = time.time()
print("Serial: ",t2-t1)
print("=======================")

@ray.remote
def det_2(mat):
	results = []
	for i in range(0,mat.shape[0]-1,2):
		for j in range(0,mat.shape[0]-1,2):
			subm = mat[np.ix_(range(i,i+2),range(j,j+2))]
			results.append(lin.det(subm))
	return results


print("=======================")
t1 = time.time()
m = div4(matriz,sizeness)
ray_ids = []
results_r = []

for mat in m:
	ray_ids.append(det_2.remote(mat))

results_r = list(itertools.chain.from_iterable(ray.get(ray_ids)))

print(len(results_r))

t2 = time.time()
print("Parallel: ",t2-t1)
print("=======================")


import code
code.interact(local=locals())
