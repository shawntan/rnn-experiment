import math
import theano
import theano.tensor as T
import numpy         as np
import utils         as U
from numpy_hinton import print_arr

def build_network(input_size,hidden_size):
	X = T.imatrix('X')
	W_input_to_hidden  = U.create_shared(U.initial_weights(input_size,hidden_size))
	W_hidden_to_output = U.create_shared(U.initial_weights(hidden_size,input_size))
	b_output = U.create_shared(U.initial_weights(input_size))

	hidden = T.nnet.sigmoid(T.dot(X,W_input_to_hidden))
	output = T.nnet.softmax(T.dot(hidden,W_input_to_hidden.T) + b_output)
	
	parameters = [W_input_to_hidden,b_output]

	return X,output,parameters

def build_error(X,output,params):
	return T.mean((X - output)**2) + sum(0.0001*T.sum(p**2) for p in params)

if __name__ == '__main__':

	X,output,parameters = build_network(8,3)
	error = build_error(X,output,parameters)
	grads = T.grad(error,wrt=parameters)
	updates = [ (W,W-grad) for W,grad in zip(parameters,grads) ]
	train = theano.function(
			inputs=[X],
			outputs=error,
			updates=updates
		)
	test = theano.function(
			inputs=[X],
			outputs=output,
		)
	data = np.eye(8,dtype=np.int32)
#	data = np.vstack((data,))
	for _ in xrange(100000):
		np.random.shuffle(data)
		print train(data)
	print_arr(test(np.eye(8,dtype=np.int32)))
	print_arr(1/(1 + np.exp(-parameters[0].get_value())),1)
