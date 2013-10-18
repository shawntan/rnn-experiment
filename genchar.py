import theano
import math
import utils
import theano.tensor as T
import numpy         as np
import utils         as U
import load_data
import sys,random
import cPickle as pickle
def make_char_outputs(data,Ws):
	result = sum(W[data[:,i]] for i,W in enumerate(Ws)) 
	return result

def make_hidden_predict_outputs(hidden_size,characters_size,
								inputs,gen_mask,
								W_i,b_i,W_o,b_o,W_pred,b_pred,W_back):
	h0 = U.create_shared(np.zeros(hidden_size))
	p0 = U.create_shared(np.zeros(characters_size))
	def step(score_t,gm,hidden_1,predict_1,W_i,b_i,W_o,b_o,W_pred,b_pred,W_back):
		hidden  = T.nnet.sigmoid(
	#			(T.dot(hidden_1,W_i) + b_i ) + \
				(1-gm) * ( T.dot(hidden_1,W_i) + b_i ) + \
				(gm  ) * ( T.dot(hidden_1,W_o) + b_o ) + \
				T.dot(predict_1,W_back) + \
				score_t
			)
		predict = T.nnet.softmax(T.dot(hidden,W_pred) + b_pred)[0]
		return hidden,predict
	[hidden_,predict_],_ = theano.scan(
			step,
			sequences     = [inputs,gen_mask],
			outputs_info  = [h0,p0],
			non_sequences = [W_i,b_i,W_o,b_o,W_pred,b_pred,W_back]
		)
	return hidden_,predict_ 

def trainer(X,Y,alpha,lr,predictions,updates,data,labels):
	data   = U.create_shared(data,  dtype=np.int8)
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
	test_model = theano.function(
			inputs  = [index_start,index_end],
			outputs = T.mean(T.neq(T.argmax(predictions, axis=1), Y)),
			givens  = {
				X:   data[index_start:index_end],
				Y: labels[index_start:index_end]
			}
		)
	return train_model,test_model

def construct_network(context,characters,hidden):
	print "Setting up memory..."
	X = T.bmatrix('X')
	Y = T.bvector('Y')
	zeros = np.zeros(characters,dtype=np.int8)
	zeros[0] = 1
	zeros[1] = 1

	alpha = T.cast(T.fscalar('alpha'),dtype=theano.config.floatX)
	lr    = T.cast(T.fscalar('lr'),dtype=theano.config.floatX)
	Ws_char_to_hidden   = [
			U.create_shared(
				U.initial_weights(characters,hidden),
				name='char[%d]'%i
			) for i in xrange(context) 
		]
	mat = Ws_char_to_hidden[0].get_value()
	mat[0] = 0
	Ws_char_to_hidden[0].set_value(mat)
	W_hidden_to_hidden_i = U.create_shared(U.initial_weights(hidden,hidden) + np.eye(hidden))
	b_hidden_i           = U.create_shared(U.initial_weights(hidden))
	W_hidden_to_hidden_o = U.create_shared(U.initial_weights(hidden,hidden) + np.eye(hidden))
	b_hidden_o           = U.create_shared(U.initial_weights(hidden))
	W_hidden_to_predict  = U.create_shared(U.initial_weights(hidden,characters))
	b_predict            = U.create_shared(U.initial_weights(characters))
	W_predict_to_hidden  = U.create_shared(U.initial_weights(characters,hidden))
	gen_weight_mask      = U.create_shared(zeros,name='mask')
	print "Constructing graph..."
	hidden_inputs  = make_char_outputs(X,Ws_char_to_hidden)
	hidden_outputs,predictions = make_hidden_predict_outputs(
			hidden,characters,
			hidden_inputs,
			gen_weight_mask[X[:,0]],
			W_hidden_to_hidden_i,
			b_hidden_i,
			W_hidden_to_hidden_o,
			b_hidden_o,
			W_hidden_to_predict,
			b_predict,
			W_predict_to_hidden			
		)


	weights = Ws_char_to_hidden + [
					W_hidden_to_hidden_i,
					b_hidden_i, 
					W_hidden_to_hidden_o,
					b_hidden_o, 
					W_hidden_to_predict,
					b_predict,
					W_predict_to_hidden
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
	return X,Y,alpha,lr,updates,predictions,weights

if __name__ == '__main__':
	context    = 1
	characters = len(load_data.chars)
	hidden     = 200
	X,Y,alpha,lr,updates,predictions,weights = construct_network(context,characters,hidden)
	p = pickle.load(open('model.data'))
	for W,pW in zip(weights,p['tunables']): W.set_value(pW)
	data,labels,start_ends = load_data.load_data(sys.argv[1])
	train,test = trainer(X,Y,alpha,lr,predictions,updates,data,labels)
	print "Done."

	lr    = 0.01
	alpha = 0.0
	decay = 0.99
	train_set = start_ends[1:]
	with open('continue','w') as f:
		for epoch in xrange(200):
			lr *= decay
			for batch,(start,end) in enumerate(train_set):
				error = train(start,end,alpha,lr)
				print "Epoch:%3d Batch:%4d Error:%.10f"%(epoch,batch,error)
			random.shuffle(train_set)
			test_error = test(*start_ends[0])
			print
			print "Test error: %.10f"%(test_error)
			print
			f.write("%.10f"%test_error)
			f.write("\n")

	pickle.dump({
		'context'    : context,
		'characters' : characters,
		'hidden'     : hidden,
		'tunables'   : [ W.get_value() for W in weights ]
	},open(sys.argv[2],'w'))
