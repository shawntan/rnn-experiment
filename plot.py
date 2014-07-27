import sys
import numpy as np
import matplotlib.pyplot as plt

data = []
min_Y, max_Y = np.inf, -np.inf
for seq_file in sys.argv[1:]:
	Y = np.array([ float(l.strip()) for l in open(seq_file) ])
	X = np.arange(Y.shape[0]) 
	min_Y,max_Y = min(np.min(Y),min_Y),max(np.max(Y),max_Y)
	data.append((X,Y))

print data
fig = plt.figure()
ax = fig.add_subplot(111)
colour = ['r','b','g']
num = 0
for X,Y in data:
	print Y
	ax.plot(X,Y,'%s-'%colour[num], linewidth=1)
	ax.axis([0,Y.shape[0],min_Y - (max_Y-min_Y)*0.1,max_Y + (max_Y-min_Y)*0.1])
	num = (num+1)%len(colour)

plt.show()
