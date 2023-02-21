import numpy as np
import matplotlib.pyplot as plt

a = np.random.normal(0, 0.5, 400000)
b = np.random.normal(0.8, 0.5, 100000)

# plt.xlim(0,1)
# plt.hist(a, bins=100, density=True, alpha=0.75, histtype='step')
# plt.hist(b, bins=100, density=True, alpha=0.75, histtype='step')
plt.hist(a, bins=100, alpha=0.75, histtype='step')
plt.hist(b, bins=100, alpha=0.75, histtype='step')
plt.savefig("Fig/testtest.png")