# Author: Danilo Cavalcanti
# Importing System Modules
import cv2 as cv
import math as m
import matplotlib.pyplot as plt
import numpy as np

# Import personal implementation library
from sobelCustom import *

# path to img
path = "./images/objects/101.pgm"
img = cv.imread(path,cv.IMREAD_GRAYSCALE)


# show image 'for visibility'
# if color img
#plt.imshow(cv.cvtColor(img,COLOR_BGR2RGB))
# if gray img
plt.imshow(img,cmap='gray')
# take away ticks from image
plt.xticks([])
plt.yticks([])

# plt.show doesn't work well with further interaction
# taken away so users manually "plt.show()" in their terminals
#plt.show()

# This is where the processing begins
x_img, y_img = createEdgeImage(np.array(img,np.float64))
# Take absolute value
x_img = np.absolute(x_img)
y_img = np.absolute(y_img)
# TODO normalize values to 0 - 255

# Convert back to Grayscale
x_img = np.uint8(x_img)
y_img = np.uint8(y_img)

# Plot side by side for comparison
plt.subplot(1,3,1), plt.imshow(img,cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2), plt.imshow(x_img,cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3), plt.imshow(y_img,cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])




import code
code.interact(local=locals())
