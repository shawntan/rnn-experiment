import numpy as np
chars = "0123456789abcdefghijklmnopqrstuvwxyz'&.$_\n"
char_map = { c: i for i,c in enumerate(chars) }

def load_data(filename):
	words = [ line.strip() for line in open(filename) ]
	X_words = ( "%s$%s\n"%(w,"_"*len(w)) for w in words )
	Y_words = ( "%s$%s\n"%("_"*len(w),w) for w in words )
	X_wordnum = np.array([ [char_map[c]] for w in X_words for c in w ],dtype=np.int8)
	Y_wordnum = np.array([ [char_map[c]] for w in Y_words for c in w ],dtype=np.int8)
	return X_wordnum, Y_wordnum

if __name__ == '__main__':
	print load_data('words')

