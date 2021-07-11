"""
Demo from opencv page, modified to not blur image beforehand
@file sobel_demo.py
@brief Sample code using Sobel and/or Scharr OpenCV functions to make a simple Edge Detector
"""
import cv2 as cv
import unittest
import sys
sys.path.append('../') # Append source-file folder
sys.path.append('../../') # Append helper function path
import detSobel as ds
import numpy as np

class Test_detSobel_behaviour(unittest.TestCase):
	def test_edge_detection_equal_to_OpenCV(self):
		test_image_file = "../../800px-1000_years_Old_Thanjavur_Brihadeeshwara_Temple_View_at_Sunrise.jpg"
		src = cv.imread(test_image_file, cv.IMREAD_GRAYSCALE)
		if src is None:
			print ('Error opening image: ' + argv[0])
			raise
		src = np.float64(src)

		### OpenCV snippet from demo
		scale = 1
		delta = 0
		ddepth = cv.CV_16S

		# Check if image is loaded properly
		
		grad_x = cv.Sobel(src, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
		grad_y = cv.Sobel(src, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
		# This normalization is applied for the way grad_x + grad_y were chosen
		# to be combined.
		grad_opencv = np.uint8((0.25*np.abs(grad_x) + 0.25*np.abs(grad_y))/2.0)

		# Exclude first and last rows and columns because
		# corner cases are excluded
		grad_opencv = np.delete(grad_opencv, [0, -1], 0)
		grad_opencv = np.delete(grad_opencv, [0, -1], 1)
		
		### Personal implementation
		grad_personal = ds.createEdgeImage(src)
		
		np.testing.assert_array_equal(grad_opencv, grad_personal)
		


if __name__=='__main__':
	unittest.main()
