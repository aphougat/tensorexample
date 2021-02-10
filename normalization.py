from numpy import array
from numpy.linalg import norm
import numpy as np
a = array([1, 2, 3])
print("vector is %s" % a)
val = np.sqrt(np.power(a[0], 2) + np.power(a[1], 2) + np.power(a[2], 2))
print("l2 norm via calculation is %s" % val)
l1 = norm(a, 1)
l2 = norm(a)
print("l1 norm via np is %s" % l1)
print("l2 norm via np is %s" % l2)
