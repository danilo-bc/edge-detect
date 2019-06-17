import time
import numpy as np
import numpy.linalg as lin
import ray
import matrixDiv as md
'''
This code tests a serial versus a parallelized implementation
of determinant calculation over submatrices in a bigger matrix.
'matriz' defines a 'big' matrix.
The serial version simply goes through every 3x3 region and applied
lin.det(), while the parallel version uses the Ray library to
give different workers for regions of the matrix.
The auxiliary library matrixDiv can divide the bigger matrix into
2, 4 and 8 submatrices to help divide the jobs for Ray.
Code must be adapted to the number of cores/cpus available in the system
'''

ray.init()

size_x = 200
size_y = 200
matriz = np.random.randint(5, size=(size_x,size_y))


print("=======================")
t1 = time.time()
mrows = matriz.shape[0]
mcols = matriz.shape[1]
results_s = np.zeros([1,(mrows-2)*(mcols-2)],dtype=np.float64)
for i in range(0,mrows-2):
	for j in range(0,mcols-2):
		subm = matriz[np.ix_(range(i,i+3),range(j,j+3))]
		results_s[0][(mcols-2)*i+j]= lin.det(subm)
print("Calculated dets: ",len(results_s[0]))
t2 = time.time()
print("Serial: ",t2-t1)
print("=======================")

@ray.remote
def det_3(mat):
	matrows = mat.shape[0]
	matcols = mat.shape[1]
	results = np.zeros([1,(matrows-2)*(matcols-2)],dtype=np.float64)
	for i in range(0,matrows-2):
		for j in range(0,matcols-2):
			subm = mat[np.ix_(range(i,i+3),range(j,j+3))]
			results[0][(matcols-2)*i+j]= lin.det(subm)
	return results


print("=======================")
t1 = time.time()
mats = md.div8(matriz)
ray_ids = []


for m in mats:
	ray_ids.append(det_3.remote(m))

results_r = np.concatenate(ray.get(ray_ids),axis=1)

print("Calculated dets: ",len(results_r[0]))

t2 = time.time()
print("Parallel: ",t2-t1)
print("=======================")

"""
import code
code.interact(local=locals())
"""