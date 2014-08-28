import theano
import theano.tensor as T
import numpy         as np
import utils         as U
from numpy_hinton import print_arr
from theano.printing import Print

W1 = U.create_shared(U.initial_weights(10,10))
W2 = U.create_shared(U.initial_weights(10,10))
b  = U.create_shared(U.initial_weights(10))
X = T.dmatrix('X')
def pair_combine(X):
	def step(i,inputs):
		length = inputs.shape[0]
		next_level = T.dot(inputs[T.arange(0,length-i-1)],W1) + T.dot(inputs[T.arange(1,length-i)],W2) + b
		next_level = next_level*(next_level > 0)
		#next_level = inputs[T.arange(0,length-i-1)] + inputs[T.arange(1,length-i)]
		#next_level = theano.printing.Print('inputs')(next_level)
		return T.concatenate([next_level,T.zeros_like(inputs[:length-next_level.shape[0]])])
	combined,_ = theano.scan(
			step,
			sequences    = [T.arange(X.shape[0])],
			outputs_info = [X],
			n_steps = X.shape[0]-1
		)
	return combined[-1,0], combined[0][:-1]
combined, pairwise = pair_combine(X)
f = theano.function(
		inputs = [X],
		outputs = [combined,pairwise]
	)
c,p = f(np.eye(10,dtype=np.float64))
print_arr(c)
print_arr(p)


