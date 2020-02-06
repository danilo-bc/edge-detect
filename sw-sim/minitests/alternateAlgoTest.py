# Author: Danilo Cavalcanti
# Importing System Modules
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import time
import timeit
import sys
sys.path.append("..")

import cProfile

# Import personal implementation library
from stoch.stochWrapper import *


## Load img and process
# img = np.array([[	0x10,0x10,	0x12],
# 				[	0x12,0x13,	0x14],
# 				[	0x00,0x00,	0x13]],np.float64)
img = np.random.randint(0,255,(3,3))
img = np.float64(img)
print('Traditional Sobel')

#cProfile.run('print(createEdgeImage(img))',sort='tottime')
print("Runtime:",timeit.timeit('print(a(im))','a = altCreateEdgeImage; im = img', globals = globals(), number = 1))

print('Alt Sobel')

#cProfile.run('print(altCreateEdgeImage(img))',sort='tottime')
print("Runtime:",timeit.timeit('print(a(im))',setup = 'a = altCreateEdgeImage; im = img', globals = globals(), number = 1))

print('Verif Sobel')
z = 8*[0]
# upper row
z[0] = img[0][0]
z[1] = img[0][1]
z[2] = img[0][2]

# middle row
z[3] = img[1][0]
#no middle pixel
z[4] = img[1][2]

# lower row
z[5] = img[2][0]
z[6] = img[2][1]
z[7] = img[2][2]
print("Runtime:",timeit.timeit('print(a(z1))',setup = 'a = verifSobelFilter; z1=z',globals = globals(), number = 1))


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

