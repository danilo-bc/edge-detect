# Author: Danilo Cavalcanti
# Importing System Modules
from bitarray import bitarray

# Importing auxiliary modules

def mux2(i1,i2,s1):
	'''Yields input 1 (i1) or i2 depending on switch 1 (s1)'''
	if(not s1):
		return i1
	else:
		return i2

def mux4(i1,i2,i3,i4,s1,s2):
	if(not s1 and not s2):
		return i1
	elif (not s1 and s2):
		return i2
	elif (s1 and not s2):
		return i3
	else:
		return i4

def sobel(z1,z2,z3,z4,z6,z7,z8,z9,r0,r1,r2,r3,r4):
	'''Implements stochastic sobel filter by Ranjbar et. al 2015'''
	# Each term equals one of the input muxes
	term1 = mux4(z2,z3,z3,z6,r0,r1) #<- different from article
	term2 = mux4(z4,z7,z7,z8,r0,r1)
	term3 = mux4(z6,z9,z9,z8,r2,r3)
	term4 = mux4(z1,z2,z1,z4,r2,r3)

	# Absolute value function done by XOR ports
	abs1 = term1^term2
	abs2 = term3^term4

	# Output pixel, z5
	return mux2(abs1,abs2,r4)
