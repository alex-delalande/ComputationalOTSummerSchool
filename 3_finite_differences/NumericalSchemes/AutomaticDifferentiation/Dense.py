import numpy as np

class denseAD(np.ndarray):
	"""
	A class for sparse forward automatic differentiation
	"""

	# Construction
	# See : https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
	def __new__(cls,value,coef=None):
		if isinstance(value,denseAD):
			assert coef is None
			return value
		obj = np.asarray(value).view(denseAD)
		shape2 = obj.shape+(0,)
		obj.coef  = np.full(shape2,0.) if coef  is None else coef
		return obj

#	def __array_finalize__(self,obj): pass

	def copy(self,order='C'):
		return denseAD(self.value.copy(order=order),self.coef.copy(order=order))

	# Representation 
	def __iter__(self):
		for value,coef in zip(self.value,self.coef):
			yield denseAD(value,coef)

	def __str__(self):
		return "denseAD"+str((self.value,self.coef))
	def __repr__(self):
		return "denseAD"+repr((self.value,self.coef))	

	# Operators
	def __add__(self,other):
		if _is_constant(other): return self.__add__(other.view(np.ndarray))
		if isinstance(other,denseAD):
			return denseAD(self.value+other.value, _add_coef(self.coef,other.coef))
		else:
			return denseAD(self.value+other, self.coef)

	def __sub__(self,other):
		if _is_constant(other): return self.__sub__(other.view(np.ndarray))
		if isinstance(other,denseAD):
			return denseAD(self.value-other.value, _add_coef(self.coef,-other.coef))
		else:
			return denseAD(self.value-other, self.coef)

	def __mul__(self,other):
		if _is_constant(other): return self.__mul__(other.view(np.ndarray))
		if isinstance(other,denseAD):
			return denseAD(self.value*other.value,_add_coef(_add_dim(other.value)*self.coef,_add_dim(self.value)*other.coef))
		elif isinstance(other,np.ndarray):
			return denseAD(self.value*other,_add_dim(other)*self.coef)
		else:
			return denseAD(self.value*other,other*self.coef)

	def __truediv__(self,other):		
		if _is_constant(other): return self.__truediv__(other.view(np.ndarray))
		if isinstance(other,denseAD):
			return denseAD(self.value/other.value,
				_add_coef(_add_dim(1/other.value)*self.coef,_add_dim(-self.value/other.value**2)*other.coef))
		elif isinstance(other,np.ndarray):
			return denseAD(self.value/other,_add_dim(1./other)*self.coef)
		else:
			return denseAD(self.value/other,(1./other)*self.coef) 

	__rmul__ = __mul__
	__radd__ = __add__
	def __rsub__(self,other): 		return -(self-other)
	def __rtruediv__(self,other): 	return denseAD(other/self.value,_add_dim(-other/self.value**2)*self.coef)

	def __neg__(self):		return denseAD(-self.value,-self.coef)

	# Math functions
	def __pow__(self,n): 	return denseAD(self.value**n, _add_dim(n*self.value**(n-1))*self.coef)
	def sqrt(self):		 	return self**0.5
	def log(self):			return denseAD(np.log(self.value), _add_dim(1./self.value)*self.coef)
	def exp(self):			return denseAD(np.exp(self.value), _add_dim(np.exp(self.value))*self.coef)
	def abs(self):			return denseAD(np.abs(self.value), _add_dim(np.sign(self.value))*self.coef)

	# Trigonometry
	def sin(self):			return denseAD(np.sin(self.value), _add_dim(np.cos(self.value))*self.coef)
	def cos(self):			return denseAD(np.cos(self.value), _add_dim(-np.sin(self.value))*self.coef)

	#Indexing
	def replace_at(self,mask,other):		
		if isinstance(other,denseAD):
			if other.size_ad==0: return self.replace_at(mask,other.view(np.array))
			elif self.size_ad==0: return other.replace_at(np.logical_not(mask),self.view(np.array))
			value,coef = np.copy(self.value), np.copy(self.coef)
			value[mask] = other.value[mask]
			coef[mask] = other.coef[mask]
			return denseAD(value,coef)
		else:
			value,coef = np.copy(self.value), np.copy(self.coef)
			value[mask]=other[mask] if isinstance(other,np.ndarray) else other
			coef[mask]=0.
			return denseAD(value,coef)

	@property
	def value(self): return self.view(np.ndarray)
	@property
	def size_ad(self):  return self.coef.shape[-1]

	def __getitem__(self,key):
		return denseAD(self.value[key], self.coef[key])

	def __setitem__(self,key,other):
		if isinstance(other,denseAD):
			if other.size_ad==0: return self.__setitem__(k,other.view(np.ndarray))
			elif self.size_ad==0: self.coef=np.zeros(other.coef.shape)
			self.value[key] = other.value
			self.coef[key] =  other.coef
		else:
			self.value[key] = other
			self.coef[key]  = 0.

	def reshape(self,shape,order='C'):
		shape2 = (shape if isinstance(shape,tuple) else (shape,))+(self.size_ad,)
		return denseAD(self.value.reshape(shape,order=order),self.coef.reshape(shape2,order=order))

	def flatten(self):	
		return self.reshape( (self.size,) )

	def broadcast_to(self,shape):
		shape2 = shape+(array.size_ad,)
		return denseAD(np.broadcast_to(array.value,shape), np.broadcast_to(array.coef,shape2) )

	@property
	def T(self):	return self if self.ndim<2 else self.transpose()
	
	def transpose(self,axes=None):
		if axes is None: axes = tuple(reversed(range(self.ndim)))
		axes2 = tuple(axes) +(self.ndim,)
		return denseAD(self.value.transpose(axes),self.coef.transpose(axes2))

	# Reductions
	def sum(self,axis=None,out=None,**kwargs):
		if axis is None: return self.flatten().sum(axis=0,out=out,**kwargs)
		out = denseAD(self.value.sum(axis,**kwargs), self.value.sum(axis,**kwargs))
		return out

#	def prod(self,axis=None,out=None,**kwargs):
#		if axis is None: return self.flatten().prod(axis=0,out=out,**kwargs)
#		result = reduce( 
#		cprod = np.cumprod(self.value,axis=axis)
#		cprod_rev = n

	def min(self,axis=0,keepdims=False,out=None):
		ai = np.expand_dims(np.argmin(self.value, axis=axis), axis=axis)
		out = np.take_along_axis(self,ai,axis=axis)
		if not keepdims: out = out.reshape(self.shape[:axis]+self.shape[axis+1:])
		return out

	def max(self,axis=0,keepdims=False,out=None):
		ai = np.expand_dims(np.argmax(self.value, axis=axis), axis=axis)
		out = np.take_along_axis(self,ai,axis=axis)
		if not keepdims: out = out.reshape(self.shape[:axis]+self.shape[axis+1:])
		return out

	def sort(self,*varargs,**kwargs):
		from . import sort
		self=sort(self,*varargs,**kwargs)


	# See https://docs.scipy.org/doc/numpy/reference/ufuncs.html
	def __array_ufunc__(self,ufunc,method,*inputs,**kwargs):

		# Return an np.ndarray for piecewise constant functions
		if ufunc in [
		# Comparison functions
		np.greater,np.greater_equal,
		np.less,np.less_equal,
		np.equal,np.not_equal,

		# Math
		np.floor_divide,np.rint,np.sign,np.heaviside,

		# 'Floating' functions
		np.isfinite,np.isinf,np.isnan,np.isnat,
		np.signbit,np.floor,np.ceil,np.trunc
		]:
			inputs_ = (a.value if isinstance(a,denseAD) else a for a in inputs)
			return super(denseAD,self).__array_ufunc__(ufunc,method,*inputs_,**kwargs)


		if method=="__call__":

			# Reimplemented
			if ufunc==np.maximum: return maximum(*inputs,**kwargs)
			if ufunc==np.minimum: return minimum(*inputs,**kwargs)

			# Math functions
			if ufunc==np.sqrt: return self.sqrt()
			if ufunc==np.log: return self.log()
			if ufunc==np.exp: return self.exp()
			if ufunc==np.abs: return self.abs()

			# Trigonometry
			if ufunc==np.sin: return self.sin()
			if ufunc==np.cos: return self.cos()

			# Operators
			if ufunc==np.add: return self.add(*inputs,**kwargs)
			if ufunc==np.subtract: return self.subtract(*inputs,**kwargs)
			if ufunc==np.multiply: return self.multiply(*inputs,**kwargs)
			if ufunc==np.true_divide: return self.true_divide(*inputs,**kwargs)


		return NotImplemented


	# Numerical 
	def solve(self):
		assert 0
		import scipy.sparse; import scipy.sparse.linalg
		return - scipy.sparse.linalg.spsolve(
        scipy.sparse.coo_matrix(self.triplets()).tocsr(),
        np.array(self).flatten()).reshape(self.shape)

	# Static methods

	# Support for +=, -=, *=, /=
	@staticmethod
	def add(a,b,out=None,where=True): 
		if out is None: return a+b #if isinstance(a,denseAD) else b+a; 
		else: result=_tuple_first(out); result[where]=a[where]+b[where]; return result

	@staticmethod
	def subtract(a,b,out=None,where=True):
		if out is None: return a-b #if isinstance(a,denseAD) else b.__rsub__(a); 
		else: result=_tuple_first(out); result[where]=a[where]-b[where]; return result

	@staticmethod
	def multiply(a,b,out=None,where=True): 
		if out is None: return a*b #if isinstance(a,denseAD) else b*a; 
		else: result=_tuple_first(out); result[where]=a[where]*b[where]; return result

	@staticmethod
	def true_divide(a,b,out=None,where=True): 
		if out is None: return a/b #if isinstance(a,denseAD) else b.__rtruediv__(a); 
		else: result=_tuple_first(out); result[where]=a[where]/b[where]; return result

	@staticmethod
	def stack(elems,axis=0):
		elems2 = tuple(denseAD(e) for e in elems)
		size_ad = max(e.size_ad for e in elems2)
		assert all((e.size_ad==size_ad or e.size_ad==0) for e in elems2)
		return denseAD( 
		np.stack(tuple(e.value for e in elems2), axis=axis), 
		np.stack(tuple(e.coef if e.size_ad==size_ad else np.zeros(e.shape+(size_ad,)) for e in elems2),axis=axis))

# -------- End of class denseAD -------

# -------- Some utility functions, for internal use -------

def _add_dim(a):		return np.expand_dims(a,axis=-1)	
def _tuple_first(a): 	return a[0] if isinstance(a,tuple) else a
def _is_constant(a):	return isinstance(a,denseAD) and a.size_ad==0

def _add_coef(a,b):
	if a.shape[-1]==0: return b
	elif b.shape[-1]==0: return a
	else: return a+b

# -------- Factory method -----

def identity(shape,shape_factor,constant=None,padding=(0,0)):
	if constant is None:
		constant = np.full(shape,0.)
	else:
		if shape is not None and shape!=constant.shape: 
			raise ValueError("identity error : incompatible shape and constant")
		else:
			shape=constant.shape

	if shape_factor!=shape[-len(shape_factor):]:
		raise ValueError("identity error : incompatible shape and shape_factor")

	ndim_elem = len(shape)-len(shape_factor)
	shape_elem = shape[:ndim_elem]
	size_elem = np.prod(shape_elem)
	size_ad = padding[0]+size_elem+padding[1]
	coef = np.full((size_elem,size_ad),0.)
	for i in range(size_elem):
		coef[i,padding[0]+i]=1.
	coef.reshape(shape_elem+(1,)*len(shape_factor)+(size_ad,))
	np.broadcast_to(coef,shape+(size_ad,))
	return denseAD(constant,coef)

# ----- Operators -----

#def add(a,other,out=None):	out=self+other; return out

# ----- Various functions, intended to be numpy-compatible ------


def maximum(a,b): 	
	from . import where
	return where(a>b,a,b)
def minimum(a,b): 	
	from . import where
	return where(a<b,a,b)


