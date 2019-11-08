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

def sobel(z1_1,
			 z2_1,
			 z3_1,
			 z4_1,
			 z6_1,
			 z7_1,
			 z8_1,
			 z9_1,
			 z1_2,
			 z2_2,
			 z3_2,
			 z4_2,
			 z6_2,
			 z7_2,
			 z8_2,
			 z9_2,
			 r0,r1,r2,r3,r4):
	'''Implements stochastic sobel filter based on Ranjbar et. al 2015
	All inputs uncorrelated'''
	# Different from the article because it implements diagonal filters
	# Each term equals one of the input muxes
	term1 = mux4(z1_1,z2_1,z2_2,z3_1,r0,r1)
	term2 = mux4(z7_1,z8_1,z8_2,z9_1,r0,r1)
	term3 = mux4(z1_2,z4_1,z4_2,z7_2,r2,r3)
	term4 = mux4(z3_2,z6_1,z6_2,z9_2,r2,r3)

	# Absolute value function done by XOR ports
	abs1 = term1^term2
	abs2 = term3^term4

	# Output pixel, z5
	return mux2(abs1,abs2,r4)

def ranjSobel(z1_1,
		  z2_1,
		  z3_1,
		  z4_1,
		  z6_1,
		  z7_1,
		  z8_1,
		  z9_1,
		  z2_2,
		  z4_2,
		  z6_2,
		  z8_2,
		  r0,r1,r2,r3,r4):
	'''Implements stochastic sobel filter based on Ranjbar et. al 2015'''
	# Different from the article because it implements diagonal filters
	# Each term equals one of the input muxes
	term1 = mux4(z1_1,z2_1,z2_2,z3_1,r0,r1)
	term2 = mux4(z7_1,z8_1,z8_2,z9_1,r0,r1)
	term3 = mux4(z1_1,z4_1,z4_2,z7_1,r2,r3)
	term4 = mux4(z3_1,z6_1,z6_2,z9_1,r2,r3)

	# Absolute value function done by XOR ports
	abs1 = term1^term2
	abs2 = term3^term4

	# Output pixel, z5
	return mux2(abs1,abs2,r4)

def diagSobel(z1_1,
		  z2_1,
		  z3_1,
		  z4_1,
		  z6_1,
		  z7_1,
		  z8_1,
		  z9_1,
		  z1_2,
		  z3_2,
		  z7_2,
		  z9_2,
		  r0,r1,r2,r3,r4):
	'''Implements stochastic sobel filter by Ranjbar et. al 2015'''
	# Each term equals one of the input muxes
	term1 = mux4(z2_1,z3_1,z3_2,z6_1,r0,r1) #<- different from article
	term2 = mux4(z4_1,z7_1,z7_2,z8_1,r0,r1)
	term3 = mux4(z6_1,z9_1,z9_2,z8_1,r2,r3)
	term4 = mux4(z2_1,z1_1,z1_2,z4_1,r2,r3)

	# Absolute value function done by XOR ports
	abs1 = term1^term2
	abs2 = term3^term4

	# Output pixel, z5
	return mux2(abs1,abs2,r4)
