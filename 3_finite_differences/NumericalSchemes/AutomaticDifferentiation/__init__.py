from . import Sparse
from . import Dense
import numpy as np

def is_adtype(t):
	return t in (Sparse.spAD, Dense.denseAD)

def is_ad(array):
	return is_adtype(type(array)) 

def is_strict_subclass(type0,type1):
	return issubclass(type0,type1) and type0!=type1

def toarray(a,array_type=np.ndarray):
	if isinstance(a,array_type): return a
	return array_type(a) if is_strict_subclass(array_type,np.ndarray) else np.array(a)

def broadcast_to(array,shape):
	if is_ad(array): return array.broadcast_to(shape)
	else: return np.broadcast_to(array,shape)

def where(mask,a,b): 
	if is_ad(b): return b.replace_at(mask,a) 
	elif is_ad(a): return a.replace_at(np.logical_not(mask),b) 
	else: return np.where(mask,a,b)

def sort(array,axis=-1,*varargs,**kwargs):
	if is_ad(array):
		ai = np.argsort(array.value,axis=axis,*varargs,**kwargs)
		return np.take_along_axis(array,ai,axis=axis)
	else:
		return np.sort(array,axis=axis,*varargs,**kwargs)

def stack(elems,axis=0):
	for e in elems:
		if is_ad(e): return type(e).stack(elems,axis)
	return np.stack(elems,axis)

def compose(a,b,shape_factor=None):
	"""Compose ad types, intended for dense a and sparse b"""
	if isinstance(a,Dense.denseAD) and (isinstance(b,Sparse.spAD) or all(isinstance(e,Sparse.spAD) for e in b)):
		elem = None
		size_factor = np.prod(shape_factor)
		if shape_factor is None:
			if not isinstance(b,Sparse.spAD):
				raise ValueError("Compose error : unspecified shape_factor")
			elem = b
		elif isinstance(b,Sparse.spAD):
			elem = b.reshape( (b.size//size_factor,)+shape_factor)
		else:
			elem = stack(e.reshape( (e.size//size_factor,)+shape_factor) for e in b)

		if elem.shape[0]!=a.size_ad:
			raise ValueError("Compose error : incompatible shapes")
		coef = np.moveaxis(a.coef,-1,0)
		first_order = sum(x*y for x,y in zip(coef,elem))
		return Sparse.spAD(a.value,first_order.coef,first_order.index)
	else:
		raise ValueError("Only Dense-Sparse composition is implemented")

def dense_eval(f,b,shape_factor):
	if isinstance(b,Sparse.spAD):
		b_dense = Dense.identity(b.shape,shape_factor,constant=b)
		return compose(f(b_dense),b,shape_factor=shape_factor)
	elif all(isinstance(e,Sparse.spAD) for e in b):
		size_factor = np.prod(shape_factor)
		size_ad_all = tuple(e.size/size_factor for e in b)
		size_ad = sum(size_ad_all)
		size_ad_cumsum = np.cumsum(size_ad_all)
		size_ad_cumsum=(0,)+size_ad_cumsum[:-1]
		size_ad_revsum = np.cumsum(reversed(size_ad_all))
		size_ad_revsum=(0,)+size_ad_revsum[:-1] 

		b_dense = stack(tuple(
			Dense.identity(e.shape,shape_factor,constant=e,padding=(padding_before,padding_after))
				for e,padding_before,padding_after in zip(b,size_ad_cumsum,size_ad_revsum) 
				))
		return compose(f(b_dense),b,shape_factor=shape_factor)
	else:
		return f(b)
