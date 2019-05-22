import numpy as np
import itertools

def OffsetToIndex(shape,offset, mode='clip', uniform=None):
	"""
	Returns the value of u at current position + offset. Also returns the coefficients and indices.
	Set padding=None for periodic boundary conditions
	"""
	ndim = len(shape)
	assert(offset.shape[0]==ndim)
	if uniform is None:
		uniform = not ((offset.ndim > ndim) and (offset.shape[-ndim:]==shape))

	grid = np.mgrid[tuple(slice(n) for n in shape)]
	grid = grid.reshape( (ndim,) + (1,)*(offset.ndim-1-ndim*int(not uniform))+shape)

	neigh = grid + (offset.reshape(offset.shape + (1,)*ndim) if uniform else offset)
	inside = np.full(neigh.shape[1:],True) # Weither neigh falls out the domain

	if mode=='wrap': # Apply periodic bc
		for coord,bound in zip(neigh,shape):
			coord %= bound
	else: #identify bad indices
		for coord,bound in zip(neigh,shape):
			inside = np.logical_and.reduce( (inside, coord>=0, coord<bound) )

	neighIndex = np.ravel_multi_index(neigh, shape, mode=mode)
	return neighIndex, inside

def TakeAtOffset(u,offset, padding=0., **kwargs):
	mode = 'wrap' if padding is None else 'clip'
	neighIndex, inside = OffsetToIndex(u.shape,offset,mode=mode, **kwargs)

	values = u.flatten()[neighIndex]
	if padding is not None:
		values[np.logical_not(inside)] = padding
	return values

def AlignedSum(u,offset,multiples,weights,**kwargs):
	"""Returns sum along the direction offset, with specified multiples and weights"""
	return sum(TakeAtOffset(u,mult*np.array(offset),**kwargs)*weight for mult,weight in zip(multiples,weights))

def Diff2(u,offset,gridScale=1.,**kwargs):
	"""Second order finite difference in the specidied direction"""
	return AlignedSum(u,offset,(1,0,-1),np.array((1,-2,1))/gridScale**2,**kwargs)

def DiffCentered(u,offset,gridScale=1.,**kwargs):
	"""Centered first order finite difference in the specified direction"""
	return AlignedSum(u,offset,(1,-1),np.array((1,-1))/(2*gridScale),**kwargs)

def DiffUpwind(u,offset,gridScale=1.,**kwargs):
	"""Upwind first order finite difference in the specified direction"""
	return AlignedSum(u,offset,(1,0),np.array((1,-1))/gridScale,**kwargs)

# -----------

def UniformGridInterpolator1D(bounds,values,mode='clip',axis=-1):
	"""Interpolation on a uniform grid. mode is in ('clip','wrap', ('fill',fill_value) )"""
	val = values.swapaxes(axis,0)
	fill_value = None
	if isinstance(mode,tuple):
		mode,fill_value = mode		
	def interp(position):
		endpoint=not (mode=='wrap')
		size = val.size
		index_continuous = (size-int(endpoint))*(position-bounds[0])/(bounds[-1]-bounds[0])
		index0 = np.floor(index_continuous).astype(int)
		index1 = np.ceil(index_continuous).astype(int)
		index_rem = index_continuous-index0
		
		fill_indices=False
		if mode=='wrap':
			index0=index0%size
			index1=index1%size
		else: 
			if mode=='fill':
				 fill_indices = np.logical_or(index0<0, index1>=size)
			index0 = np.clip(index0,0,size-1) 
			index1 = np.clip(index1,0,size-1)
		
		index_rem = index_rem.reshape(index_rem.shape+(1,)*(val.ndim-1))
		result = val[index0]*(1.-index_rem) + val[index1]*index_rem
		if mode=='fill': result[fill_indices] = fill_value
		result = np.moveaxis(result,range(position.ndim),range(-position.ndim,0))
		return result
	return interp

def UniformGridInterpolator(bounds,values,mode='clip',axes=None):
	"""Assumes 'ij' indexing by defautl. Use axes=(1,0) for 'xy' """
	ndim_interp = len(bounds)
	if axes is None:
		axes = tuple(range(-ndim_interp,0))
	val = np.moveaxis(values,axes,range(ndim_interp))

	fill_value = None
	if isinstance(mode,tuple):
		mode,fill_value = mode

	def interp(position):
		endpoint=not (mode=='wrap')
		ndim_val = val.ndim - ndim_interp
		shape_to_point = (ndim_interp,)+(1,)*ndim_val
		shape = np.array(val.shape[:ndim_interp]).reshape(shape_to_point)
		bounds0,bounds1 = (np.array(bounds)[:,i].reshape(shape_to_point) for i in (0,-1))
		index_continuous = (shape-int(endpoint)) * (position-bounds0) / (bounds1-bounds0)
		index0 = np.floor(index_continuous).astype(int)
		index1 = np.ceil(index_continuous).astype(int)
		index_rem = index_continuous-index0

		fill_indices = False
		for i,s in enumerate(shape.flatten()):
			if mode=='wrap':
				index0[i]=index0[i]%s
				index1[i]=index1[i]%s
			else: 
				if mode=='fill':
					fill_indices = np.logical_or.reduce((index0<0, index1>=s))
				index0[i] = np.clip(index0[i],0,s-1) 
				index1[i] = np.clip(index1[i],0,s-1)

		def prod(a): 
			result=a[0]; 
			for ai in a[1:]: result*=ai; 
			return result

		index_rem = index_rem.reshape(index_rem.shape+(1,)*ndim_val)		 
		result = sum( #Worryingly, priority rules of __rmul__ where not respected here ?
			prod(tuple( (1.-r) if m else r for m,r in zip(mask,index_rem)) ) *
			val[ tuple(np.where(mask,index0,index1)) ]
			for mask in itertools.product((True,False),repeat=ndim_interp))


		if mode=='fill': result[fill_indices] = fill_value
		result = np.moveaxis(result,range(position.ndim-1),range(-position.ndim+1,0))
		return result
	return interp









