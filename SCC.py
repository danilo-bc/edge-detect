from bitarray import bitarray
# Currently unused in the project

def SCC(X,Y):
	'''Calculated Stochastic Circuit Correlation between X and Y'''
	# overlapping 1's
	a = (X&Y).count()
	# overlapping 1's of X and 0's of Y
	b = (X&~Y).count()
	# overlapping 0's of X and 1's of Y
	c = ((~X)&Y).count()
	# overlapping 0's
	d = (~(X|Y)).count()
	if(a*d>b*c):
		return (a*d-b*c)/(len(X)*np.min([a+b,a+c])-(a+b)*(a+c))
	else:
		return (a*d-b*c)/((a+b)*(a+c)-len(X)*np.max([a-d,0]))