import unittest
import sys
sys.path.append('../') # Append source-file folder
sys.path.append('../../') # Append to solve module loading
import ray_stochWrapper as rsw
import stochWrapper as sw
import numpy as np

import ray

class Test_ray_stochWrapper_behaviour(unittest.TestCase):
    def setUp(self):
        ray.init(local_mode=True)

    def tearDown(self):
        ray.shutdown()

    def test_ray_version_outputs_same_result_as_regular(self):
        src, edges_ref = sw.detectAndShow('scaled_down_sample.jpg',show=False)
        src, edges_ray = rsw.ray_detectAndShow('scaled_down_sample.jpg',show=False)
        np.testing.assert_array_equal(edges_ref, edges_ray)

if __name__=='__main__':
    unittest.main()