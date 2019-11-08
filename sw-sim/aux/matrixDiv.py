import numpy as np
'''Auxiliary library to subdivide a matrix in groupings of
3x3 elements to speedup Sobel algorithm simulation'''

def div2(mat):
	rows = mat.shape[0]
	cols = mat.shape[1]
	# Half values truncated for use in range()
	half_rows = int(rows/2)
	half_cols = int(cols/2)
	# Divided quadrants
	Q = 2*[0]
	# ix_ works like "(rows,cols)"
	# Values adapted for range()
	Q[0] = mat[np.ix_(range(0,rows),range(0,half_cols+1))]
	Q[1] = mat[np.ix_(range(0,rows),range(half_cols-1,cols))]
	return Q

def div4(mat):
	rows = mat.shape[0]
	cols = mat.shape[1]
	# Half values truncated for use in range()
	half_rows = int(rows/2)
	half_cols = int(cols/2)
	# Divided quadrants
	Q = 4*[0]
	# ix_ works like "(rows,cols)"
	# Values adapted for range()
	Q[0] = mat[np.ix_(range(0,half_rows+1),range(0,half_cols+1))]
	Q[1] = mat[np.ix_(range(0,half_rows+1),range(half_cols-1,cols))]
	Q[2] = mat[np.ix_(range(half_rows-1,rows),range(0,half_cols+1))]
	Q[3] = mat[np.ix_(range(half_rows-1,rows),range(half_cols-1,cols))]

	return Q

def div8(mat):
	Q4 = div4(mat)
	Q = 8*[0]
	Q[0],Q[1] = div2(Q4[0])
	Q[2],Q[3] = div2(Q4[1])
	Q[4],Q[5] = div2(Q4[2])
	Q[6],Q[7] = div2(Q4[3])

	return Q