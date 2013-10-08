import theano
import math
import utils
import theano.tensor as T
import numpy         as np
import utils         as U
import load_data
import sys
import cPickle as pickle
def make_char_outputs(data,Ws,b=None):
	result = sum(W[data[:,i]] for i,W in enumerate(Ws))
	if b: result += b
	return result

def make_hidden_outputs(inputs,W,hidden):
	h0 = U.create_shared(np.zeros((hidden,)))
	def step(score_t,self_tm1,W):
		return T.nnet.sigmoid(score_t + T.dot(self_tm1,W))
		#return T.tanh(score_t + T.dot(self_tm1,W))
	activation_probs,_ = theano.scan(
			step,
			sequences     = inputs,
			outputs_info  = h0,
			non_sequences = W
		)
	return activation_probs

def make_predictions(inputs,W,b,more=None):
	return T.nnet.softmax(b + T.dot(inputs,W)+more)

def trainer(X,Y,alpha,lr,predictions,updates,data,labels):
	data   = U.create_shared(data,dtype=np.int8)
	labels = U.create_shared(labels,dtype=np.int8)
	index_start = T.lscalar('start')
	index_end   = T.lscalar('end')
	print "Compiling function..."
	train_model = theano.function(
			inputs  = [index_start,index_end,alpha,lr],
			outputs = T.mean(T.neq(T.argmax(predictions, axis=1), Y)),
			updates = updates,
			givens  = {
				X:   data[index_start:index_end],
				Y: labels[index_start:index_end]
			}
		)
	return train_model

def construct_network(context,characters,hidden):
	print "Setting up memory..."
	X = T.bmatrix('X')
	Y = T.bvector('Y')
	alpha = T.cast(T.fscalar('alpha'),dtype=theano.config.floatX)
	lr    = T.cast(T.fscalar('lr'),dtype=theano.config.floatX)
	Ws_char_to_hidden   = [ U.create_shared(U.initial_weights(characters,hidden),name='char[%d]'%i) for i in xrange(context) ]
	Ws_char_to_predict  = [ U.create_shared(U.initial_weights(characters,characters),name='char[%d]'%i) for i in xrange(context) ]
	b_hidden            = U.create_shared(U.initial_weights(hidden))
	W_hidden_to_hidden  = U.create_shared(U.initial_weights(hidden,hidden))
	W_hidden_to_predict = U.create_shared(U.initial_weights(hidden,characters))
	b_predict           = U.create_shared(U.initial_weights(characters))
	print "Constructing graph..."
	hidden_inputs  = make_char_outputs(X,Ws_char_to_hidden,b_hidden)
	hidden_outputs = make_hidden_outputs(hidden_inputs,W_hidden_to_hidden,hidden)
	predictions    = make_predictions(
			hidden_outputs,
			W_hidden_to_predict,
			b_predict,
			make_char_outputs(X,Ws_char_to_predict)
		)
	
	weights = Ws_char_to_hidden +\
			  Ws_char_to_predict + [
					b_hidden, 
					W_hidden_to_hidden,
					W_hidden_to_predict,
					b_predict
				]
	cost    = -T.mean(T.log(predictions)[T.arange(Y.shape[0]),Y])
	gparams =  T.grad(cost,weights)

	deltas  = [ U.create_shared(np.zeros(w.get_value().shape)) for w in weights ]
	updates = [
				( param, param - ( alpha * delta + gparam * lr ) )
					for param,delta,gparam in zip(weights,deltas,gparams)
			] + [
				( delta, alpha * delta + gparam * lr)
					for delta,gparam in zip(deltas,gparams)
			]
	return X,Y,alpha,lr,updates,predictions

if __name__ == '__main__':
	context    = 1
	characters = len(load_data.chars)
	hidden     = 500
	X,Y,alpha,lr,updates,predictions = construct_network(context,characters,hidden)
	data,labels,start_ends = load_data.load_data(sys.argv[1])
	train = trainer(X,Y,alpha,lr,predictions,updates,data,labels)
	print "Done."

	lr    = 0.1
	alpha = 0.5
	decay = 0.95
	for epoch in xrange(500):
		lr *= decay
		for batch,(start,end) in enumerate(start_ends):
			error = train(start,end,alpha,lr)
			print "Epoch:%3d Batch:%4d Error:%.10f"%(epoch,batch,error)
	pickle.dump({
		'context': context,
		'characters': characters,
		'hidden': hidden,
		'tunables': [ W.get_value() for W in tunables ]
	},open(sys.argv[2],'w'))
