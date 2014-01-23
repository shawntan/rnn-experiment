import theano
import math
import utils
import theano.tensor as T
import numpy         as np
import utils         as U
import load_data
import sys,random
import cPickle as pickle

def make_hidden(hidden_size,add_ins,mult_ins,Wf,fW,b):
	h0 = U.create_shared(np.zeros(hidden_size))
	def step(add_in,mult_in,hidden_1,Wf,fW,b):
		mult_W = T.dot(Wf * mult_in,fW)
		hidden_score = add_in + T.dot(hidden_1,mult_W) + b
		return T.nnet.sigmoid(hidden_score)
	hidden,_ = theano.scan(
			step,
			sequences     = [add_ins,mult_ins],
			outputs_info  = [h0],
			non_sequences = [Wf,fW,b]
		)
	return hidden


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
	print "Done."
	return train_model,test_model

def construct_network(context,characters,hidden,mult_hidden):
	print "Setting up memory..."
	X = T.bvector('X')
	Y = T.bvector('Y')
	alpha = T.cast(T.fscalar('alpha'),dtype=theano.config.floatX)
	lr    = T.cast(T.fscalar('lr'),   dtype=theano.config.floatX)
	
	print "Initialising weights..."
	W_char_hidden    = U.create_shared(U.initial_weights(characters,hidden))
	f_char_hidden    = U.create_shared(U.initial_weights(characters,mult_hidden))
	b_hidden         = U.create_shared(U.initial_weights(hidden))
	Wf_hidden        = U.create_shared(U.initial_weights(hidden,mult_hidden))
	fW_hidden        = U.create_shared(U.initial_weights(mult_hidden,hidden))
	W_hidden_predict = U.create_shared(U.initial_weights(hidden,characters))
	b_predict        = U.create_shared(U.initial_weights(characters))

	print "Constructing graph..."
	hidden = make_hidden(
			hidden,
			W_char_hidden[X],
			f_char_hidden[X],
			Wf_hidden,
			fW_hidden,
			b_hidden
		)
	predictions = T.nnet.softmax(T.dot(hidden,W_hidden_predict) + b_predict)
	weights = [
			W_char_hidden,
			f_char_hidden,
			b_hidden,
			Wf_hidden,
			fW_hidden,
			W_hidden_predict,
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
	return X,Y,alpha,lr,updates,predictions,weights

if __name__ == '__main__':
	context     = 1
	characters  = len(load_data.chars)
	hidden      = 500
	mult_hidden = 500
	X,Y,alpha,lr,updates,predictions,weights = construct_network(context,characters,hidden,mult_hidden)
	#p = pickle.load(open('model.data'))
	#for W,pW in zip(weights,p['tunables']): W.set_value(pW)
	data,labels,start_ends = load_data.load_data(sys.argv[1])
	train,test = trainer(X,Y,alpha,lr,predictions,updates,data,labels)

	lr    = 1
	alpha = 0.5
	decay = 0.95
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
