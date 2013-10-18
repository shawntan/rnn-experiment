import numpy as np
import random
import math
chars = " _$0123456789abcdefghijklmnopqrstuvwxyz'&."
char_map = { c: i for i,c in enumerate(chars) }
def load_data(filename,batch_size=100):
	words = [ line.strip() for line in open(filename) ]
	random.shuffle(words)
	batches = int(math.ceil(len(words)/float(batch_size)))
	X_words = [ "%s%s "%(w,"_"*len(w)) for w in words ]
	Y_words = [ "%s%s "%(w,w) for w in words ]

	X_wordnum = np.array([ char_map[c] for w in X_words for c in w ], dtype=np.int8)
	Y_wordnum = np.array([ char_map[c] for w in Y_words for c in w ], dtype=np.int8)
	lengths = [ sum(len(w) for w in words[i*batch_size:(i+1)*batch_size])
						for i in xrange(batches) ]
	start_ends = []
	start = 0
	for l in lengths:
		start_ends.append((start,start+l))
		start += l
	
	return X_wordnum, Y_wordnum, start_ends



if __name__ == '__main__':
	print load_data('words')
