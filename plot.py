import sys
import numpy as np
import matplotlib.pyplot as plt

data = []
min_Y, max_Y = np.inf, -np.inf
filenames = sys.argv[1:]
shortest = np.inf
for seq_file in filenames:
	Y = np.array([ float(l.strip()) for l in open(seq_file) ])
	if len(Y) < shortest: shortest = len(Y)
	X = np.arange(Y.shape[0]) 
	min_Y,max_Y = min(np.min(Y),min_Y),max(np.max(Y),max_Y)
	data.append((X,Y))

fig = plt.figure()
ax = fig.add_subplot(111)
colour = ['r','b','g','y']
num = 0
plots = []
for (X,Y),name in zip(data,filenames):
	X,Y=X[:shortest],Y[:shortest]
	plots.append(plt.plot(X,Y,'%s-'%colour[num], linewidth=1,label=name))
	plt.axis([0,Y.shape[0],min_Y - (max_Y-min_Y)*0.1,max_Y + (max_Y-min_Y)*0.1])
	num = (num+1)%len(colour)

print plots,filenames
plt.legend(filenames)
plt.show()
