import cv2 as cv
import math as m
import matplotlib.pyplot as plt
import numpy as np

# path to img
path = "redim.jpg"
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

# Initialize empty matrix to receive Gradient
newimg = np.array(np.zeros(img.shape) ,dtype='uint8')


import code
code.interact(local=locals())
