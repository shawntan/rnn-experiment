import theano
import math
import utils
import theano.tensor as T
import numpy         as np
import utils         as U
import load_data
import sys
def make_hidden_inputs(data,Ws,b):
	return sum(W[data[:,i]] for i,W in enumerate(Ws)) + b

	

def make_hidden_outputs(inputs,W,hidden):
	h0 = U.create_shared(np.zeros((hidden,)))
	def step(score_t,self_tm1,W):
		return T.nnet.sigmoid(score_t + T.dot(self_tm1,W))
	activation_probs,_ = theano.scan(
			step,
			sequences     = inputs,
			outputs_info  = h0,
			non_sequences = W
		)
	return activation_probs

def make_predictions(inputs,W,b):
	return T.nnet.softmax(b + T.dot(inputs,W))

def trainer(X,Y,predictions,tunables,data,labels):
	lr      = 0.1
	cost    = -T.mean(T.log(predictions)[T.arange(Y.shape[0]),Y])
	gparams =  T.grad(cost,tunables)
	updates =  [ (param, param - gparam * lr) for param,gparam in zip(tunables,gparams) ]
	print "Compiling function..."
	train_model = theano.function(
			inputs  = [],
			outputs = T.mean(T.neq(T.argmax(predictions, axis=1), Y)),
			updates = updates,
			givens  = {
				X: data,
				Y: labels,
			}
		)
	return train_model

def construct_network(context,characters,hidden):
	print "Setting up memory..."
	X = T.bmatrix('X')
	Y = T.bvector('Y')
	Ws_char_to_hidden   = [ U.create_shared(U.initial_weights(characters,hidden),name='char[%d]'%i) for i in xrange(context) ]
	b_hidden            = U.create_shared(U.initial_weights(hidden))
	W_hidden_to_hidden  = U.create_shared(U.initial_weights(hidden,hidden))
	W_hidden_to_predict = U.create_shared(U.initial_weights(hidden,characters))
	b_predict           = U.create_shared(U.initial_weights(characters))
	print "Constructing graph..."
	hidden_inputs  = make_hidden_inputs(X,Ws_char_to_hidden,b_hidden)
	hidden_outputs = make_hidden_outputs(hidden_inputs,W_hidden_to_hidden,hidden)
	predictions    = make_predictions(hidden_outputs,W_hidden_to_predict,b_predict)
	tunables = Ws_char_to_hidden + [
			b_hidden, 
			W_hidden_to_hidden,
			W_hidden_to_predict,
			b_predict
		]
	return X,Y,tunables,predictions

if __name__ == '__main__':
	context    = 1
	characters = len(load_data.chars)
	hidden     = 100
	X,Y,tunables,predictions = construct_network(context,characters,hidden)
	data,labels = load_data.load_data(sys.argv[1])
	train = trainer(X,Y,predictions,tunables,data,labels)
	print "Done."
	for i in xrange(10): print train()
