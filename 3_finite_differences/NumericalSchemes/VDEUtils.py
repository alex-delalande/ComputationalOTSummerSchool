import numpy as np
from NumericalSchemes import FileIO
#FileVDE_binary_dir = None

def FlattenSymmetricMatrix(a):
	"""
		Extract the lower triangular part of a (symmetric) matrix, put in a vector
	"""
	m,n = a.shape[:2]
	if m!=n:
		raise ValueError("VDEUtils.FlattenSymmetricMatrix error: matrix is not square")
	return np.array([a[i,j] for i in range(n) for j in range(i+1)])

def V2Sym(a):
	d = a.shape[0]
	m = int(np.floor(np.sqrt(2*d)))
	if d!=m*(m+1)//2:
		raise ValueError("V2Sym error: first dimension not of the form d(d+1)/2")
	def index(i,j):
		a,b=np.maximum(i,j),np.minimum(i,j)
		return a*(a+1)//2+b
	return np.array([ [a[index(i,j)] for i in range(m)] for j in range(m)])

def Decomposition(a):
	"""
		Call the FileVDZ library to decompose the provided tensor field.
	"""
	if FileVDE_binary_dir is None:
		raise ValueError("VDEUtils.Decomposition error : path to FileVDE binaries not specified")
	vdeIn ={'tensors':np.moveaxis(FlattenSymmetricMatrix(a),0,-1)}
	vdeOut = FileIO.WriteCallRead(vdeIn, "FileVDE", FileVDE_binary_dir)
	return np.moveaxis(vdeOut['weights'],-1,0),np.moveaxis(vdeOut['offsets'],[-1,-2],[0,1])