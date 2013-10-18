import sys
import numpy as np
import matplotlib.pyplot as plt

Y = np.array([ float(l.strip()) for l in open(sys.argv[1]) ])
X = np.arange(Y.shape[0]) 
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(X,Y,'r-', linewidth=1)
ax.axis([0,200,0,1])
plt.show()
