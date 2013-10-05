import theano
import math
import utils
import theano.tensor as T
import numpy         as np
import utils         as U
CHARACTERS = 86
CONTEXT    = 5
HIDDEN     = 20

def make_hidden_inputs(data,Ws,b):
	return sum(W[data[:,i]] for i,W in enumerate(Ws)) + b

	

def make_hidden_outputs(inputs,W):
	h0 = U.create_shared(np.zeros((HIDDEN,)))
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



if __name__ == '__main__':
	print "Setting up memory..."
	X = T.bmatrix('X')
	Y = T.bvector('Y')
	Ws_char_to_hidden   = [ U.create_shared(U.initial_weights(CHARACTERS,HIDDEN),name='char[%d]'%i) for i in xrange(CONTEXT) ]
	b_hidden            = U.create_shared(U.initial_weights(HIDDEN))
	W_hidden_to_hidden  = U.create_shared(U.initial_weights(HIDDEN,HIDDEN))
	W_hidden_to_predict = U.create_shared(U.initial_weights(HIDDEN,CHARACTERS))
	b_predict           = U.create_shared(U.initial_weights(CHARACTERS))
	tunables = Ws_char_to_hidden + [
			b_hidden, 
			W_hidden_to_hidden,
			W_hidden_to_predict,
			b_predict
		]

	print "Constructing graph..."
	hidden_inputs  = make_hidden_inputs(X,Ws_char_to_hidden,b_hidden)
	hidden_outputs = make_hidden_outputs(hidden_inputs,W_hidden_to_hidden)
	predictions    = make_predictions(hidden_outputs,W_hidden_to_predict,b_predict)

	data = U.create_shared(np.array(
		[[0,1,2,3,4],
		 [1,2,3,4,5],
		 [2,3,4,5,6]],
		),dtype=np.int8)
	labels = U.create_shared(np.array([5,6,7]),dtype=np.int8)
	print "Compiling function..."
	train = trainer(X,Y,predictions,tunables,data,labels)
	print "Done."
	for i in xrange(10): print train()
