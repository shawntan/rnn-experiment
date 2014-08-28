from itertools import izip

def updates(parameters,gradients,rho,eps):
	gradients_sq = [ U.create_shared(np.zeros(p.get_value().shape)) for p in parameters ]
	deltas_sq    = [ U.create_shared(np.zeros(p.get_value().shape)) for p in parameters ]

	gradients_sq_new = [ rho*g_sq + (1-rho)*(g**2) for g_sq,g in izip(gradients_sq,gradients) ]
	deltas = [ (T.sqrt(d_sq+eps)/T.sqrt(g_sq+eps))*grad for d_sq,g_sq,grad in izip(deltas_sq,gradients_sq_new,gradients) ]
	deltas_sq_new = [ rho*d_sq + (1-rho)*(d**2) for d_sq,d in izip(deltas_sq,deltas) ]

	gradient_sq_updates = zip(gradients_sq,gradients_sq_new)
	deltas_sq_updates = zip(deltas_sq,deltas_sq_new)
	parameters_updates = [ (p,p - d) for p,d in izip(



