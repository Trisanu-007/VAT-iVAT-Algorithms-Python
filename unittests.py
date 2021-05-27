import unittest
import numpy as np
from distance2 import distance2
from numpy import testing


# class Testcases(unittest.TestCase):

#     def test_distance2(self):
#         print(distance2(np.array([[5, 2], [7, 3], [8, 1]]), np.array(
#             [[5, 2], [7, 3], [8, 1]])))
#         self.assertAlmostEqual(
#             distance2(np.array([[5, 2], [7, 3], [8, 1]]), np.array([[5, 2], [7, 3], [8, 1]])), np.array([[0, 2.236, 3.162], [2.236, 0, 2.236], [3.162, 2.236, 0]]), "Passed")


# if __name__ == "__main__":
#     unittest.main()

# if __name__ == "__main__":
testing.assert_almost_equal(distance2(np.array([[5, 2], [7, 3], [8, 1]]), np.array(
    [[5, 2], [7, 3], [8, 1]])), np.array([[0, 2.236, 3.162], [2.236, 0, 2.236], [3.162, 2.236, 0]]), decimal=3, err_msg="Not Equal!")
