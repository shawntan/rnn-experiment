import theano
import theano.tensor as T
import numpy         as np
import utils         as U

switch = T.scalar('switch')
A = U.create_shared(np.eye(8))
weights = U.create_shared(U.initial_weights(8,3))
hidden  = T.nnet.sigmoid(T.dot(A,weights))
recon   = T.nnet.softmax(switch*T.dot(hidden,weights.T))

cost     = T.sum((A-recon)**2)
gradient = T.grad(cost,wrt=weights)

updates  = [ (weights, weights - gradient) ]

print "Compiling..."
f = theano.function(
		inputs  = [switch],
		updates = updates,
		outputs = cost
	)
print "Done."
for _ in xrange(1000000): print f(0)

