# Author: Danilo Cavalcanti
# Importing System Modules
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import time

# Import personal implementation library
from stochWrapper import *
import detSobel as detS

ray.init()

print('''------------------------------------------------------------------
------------------------------------------------------------------
Sucessfully loaded personal Sobel Filter implementation
How to demo:
Use functions 'detectAndShow(image)' or 'detectAndWritePGM(image)'
Example: src,edges = rayDetectAndShow('./images/aerial/school.pgm')
Result: photo printed on screen with edges detected,
base image returned as 'src' and
edge image returned into 'edges' variable as numpy matrix
------------------------------------------------------------------
------------------------------------------------------------------''')

random.seed(32)
lfsrSize = 8
half = 127
auxStr = '{:0'+str(lfsrSize)+'b}'
#5 numbers from 16x16 hadamard matrix that have SCC = 0
#r_had =[bitarray('1001100110011001'),bitarray('1111000011110000'),
#		bitarray('1010010110100101'),bitarray('1100001111000011'),
#		bitarray('1001011010010110')]

r_had =[bitarray('10011001'),
		bitarray('11110000'),
		bitarray('10100101'),
		bitarray('11000011'),
		bitarray('10010110')]

# 8 random streams for all but center pixel
# 4 random streams for copies of
# - z2, z4, z6 and z8 for vertical/horizontal
# - z1, z3, z7 and z9 for diagonal sobel
rng_z_1_rand = 8*[0]
rng_z_1_verilog = [bitarray('00010011'),
				   bitarray('11101101'),
				   bitarray('00110110'),
				   bitarray('00100101'),
				   bitarray('01001101'),
				   bitarray('10110010'),
				   bitarray('11100110'),
				   bitarray('00111100')]
for i in range(8):
	rng_z_1_rand[i] = bitarray(auxStr.format(random.getrandbits(lfsrSize)))

img = np.array([[	54,53,		44],
				[	127,130,	149],
				[	130,131,	139]],np.float64)
#img = np.float64(loadHex('./hw-sim/stochastic-manual/square3.txt'))
for i in range (1):
	#img = np.float64(np.random.randint(255,size=(3,3)))
	print(img)
	print("-------------------------------")
	resultD = np.uint8(detS.sobelFilter(img))
	#resultSrandalt = sobelFilter(img,r_had,rng_z_1_rand)
	resultSverilog = sobelFilter(img,r_had,rng_z_1_verilog)

	#img = np.rot90(img)
	print('Resultado det:        ', int(resultD))
	#print('Resultado rand alt:   ', hex(resultSrandalt))
	print('Resultado Verilog:   ',  hex(resultSverilog))
	#print('Resultado Erro:   ',  hex(resultError))




# real application test:
"""

print("=======================")
t1 = time.time()

src,edges = rayDetectAndShow('./images/aerial/school.pgm')
#saveToHex('school_edges_sw.txt',edges)
t2 = time.time()
print("Tempo: ",t2-t1)
print("=======================")
img1 = importHex('school_edges_hw.txt')
"""

# import code
# code.interact(local=locals())

