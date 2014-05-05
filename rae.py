import theano
import math
import utils
import theano
import theano.tensor as T
import numpy         as np
import utils         as U
import load_data
import sys,random
import cPickle as pickle
import numpy as np
from numpy_hinton import print_arr

def unroll(final_rep,W2_m,b2_m,W2_i,b2_i,n_steps):
	def step(curr_rep,W2_m,b2_m,W2_i,b2_i):
		next_rep  = T.dot(curr_rep,W2_m) + b2_m
		input_rep = T.dot(curr_rep,W2_i) + b2_i
		return next_rep,input_rep
	[_,recon],_ = theano.scan(
			step,
			outputs_info = [final_rep,None],
			non_sequences  = [W2_m,b2_m,W2_i,b2_i],
			n_steps = n_steps
		)
	return recon


def make_rae(inputs,W1_m,W1_i,b_h,i_h,W2_m,b2_m,W2_i,b2_i):
	def step(
			inputs,
			hidden_1,
			W1_m,W1_i,b_h,W2_m,b2_m,W2_i,b2_i):
		#		hidden = T.nnet.sigmoid(
		hidden = T.tanh(
				T.dot(hidden_1,W1_m) +\
				T.dot(inputs,W1_i) +\
				b_h
			)
		reproduction_m = T.dot(hidden,W2_m) + b2_m
		reproduction_i = T.dot(hidden,W2_i) + b2_i
		return hidden,reproduction_m,reproduction_i
	[hidden_,reproduction_m_,reproduction_i_],_ = theano.scan(
			step,
			sequences     = [inputs],
			outputs_info  = [i_h,None,None],
			non_sequences = [W1_m,W1_i,b_h,W2_m,b2_m,W2_i,b2_i]
			)
	return hidden_,reproduction_m_,reproduction_i_

def build_network(input_size,hidden_size):
	X = T.dmatrix('X')
	W_input_to_hidden  = U.create_shared(U.initial_weights(input_size,hidden_size))
	W_hidden_to_hidden = U.create_shared(U.initial_weights(hidden_size,hidden_size))
	b_hidden = U.create_shared(U.initial_weights(hidden_size))
#	initial_hidden = U.create_shared(U.initial_weights(hidden_size))
	initial_hidden = U.create_shared(np.zeros(hidden_size))

	W_hidden_to_hidden_reproduction = U.create_shared(U.initial_weights(hidden_size,hidden_size))
	b_hidden_reproduction           = U.create_shared(U.initial_weights(hidden_size))
	W_hidden_to_input_reproduction  = U.create_shared(U.initial_weights(hidden_size,input_size))
	b_input_reproduction            = U.create_shared(U.initial_weights(input_size))
	parameters = [
			W_input_to_hidden,
			W_hidden_to_hidden,
			b_hidden,
			initial_hidden,
			W_hidden_to_hidden_reproduction,
			b_hidden_reproduction,
			W_hidden_to_input_reproduction,
			b_input_reproduction
		]

	hidden, hidden1_reproduction, input_reproduction = make_rae(
			X,
			W_input_to_hidden,
			W_hidden_to_hidden,
			b_hidden,
			initial_hidden,
			W_hidden_to_hidden_reproduction,
			b_hidden_reproduction,
			W_hidden_to_input_reproduction,
			b_input_reproduction
		)

	unrolled = unroll(
			hidden[-1],
			W_hidden_to_hidden_reproduction,
			b_hidden_reproduction,
			W_hidden_to_input_reproduction,
			b_input_reproduction,
			hidden.shape[0]
		)

	return X,parameters,hidden,hidden1_reproduction,input_reproduction,unrolled


def build_error(X,hidden,hidden1_reproduction,input_reproduction):
	input_reproduction_sqerror  = T.sum((X - input_reproduction)**2)
	hidden_reproduction_sqerror = T.sum((hidden[:-1] - hidden1_reproduction[1:])**2)
	return input_reproduction_sqerror + hidden_reproduction_sqerror

if __name__ == '__main__':
	X,parameters,hidden,hidden1_reproduction,input_reproduction,unrolled = build_network(10,10)
	f = theano.function(
			inputs  = [X],
			outputs = [hidden,hidden1_reproduction,input_reproduction,unrolled]
		)

	error = build_error(X,hidden,hidden1_reproduction,input_reproduction)
	gradients = T.grad(error,wrt=parameters)

	updates = [ (p, p - 0.001*g) for p,g in zip(parameters,gradients) ]
	train = theano.function(
			inputs = [X],
			updates = updates,
			outputs = error
		)
	for _ in xrange(10000): print train(np.eye(10))

	hidden, hidden_rep, input_rep, unrlld  = f(np.eye(10))

	print_arr(hidden)
	print_arr(input_rep)
	print_arr(unrlld)
#	print_arr(unrlld,hidden)
#	print_arr(parameters[4].get_value())
