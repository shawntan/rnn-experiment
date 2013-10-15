import theano
import math
import utils
import theano.tensor as T
import numpy         as np
import utils         as U
from load_data import *
import sys
import cPickle as pickle
from genchar import construct_network

def load_network(filename):
	p = pickle.load(open(filename))
	X,Y,alpha,lr,updates,predictions,weights = construct_network(p['context'],p['characters'],p['hidden'])
	for W,pW in zip(weights,p['tunables']): W.set_value(pW)
	return X,predictions
if __name__ == '__main__':
	w = "%s%s "%(sys.argv[2],"_"*len(sys.argv[2]))
	instance = np.array([ [char_map[c]] for c in w ], dtype=np.int8)
	X,predictions = load_network(sys.argv[1])
	predict = theano.function(
		inputs  = [X],
		outputs = T.argmax(predictions, axis=1)
	)
	print ''.join( chars[n] for n in predict(instance) )
