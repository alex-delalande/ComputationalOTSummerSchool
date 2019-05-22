import numpy as np
import scipy.sparse as sp

from NumericalSchemes import Selling
from NumericalSchemes import LinearParallel as LP

diff = np.zeros((3,3,2,2))
diff[:,:,0,0]=1
diff[:,:,1,1]=1

"""
 Constructs a linear operator sparse matrix, given as input 
- an array of psd matrices, denoted diff
- an array of vectors (optionnal), denoted omega
- an array of scalars (optionnal),

additional parameters
- a grid scale 
- boundary conditions, possibly axis by axis 
	('Periodic', 'Reflected', 'Neumann', 'Dirichlet') 
- divergence for or not

Returns : a list of triplets, for building a coo matrix
"""
def OperatorMatrix(diff,omega=None,mult=None, \
		gridScale=1,
		boundaryConditions='Periodic', 
		divergenceForm=False,
		intrinsicDrift=False):
	
	# ----- Get the domain shape -----
	bounds = diff.shape[2:]
	dim = len(bounds)
	if isinstance(boundaryConditions,str):
		boundaryConditions = np.full( (dim,2), boundaryConditions)
	elif len(boundaryConditions)!=dim:
		raise ValueError("""OperatorMatrix error : 
		inconsistent boundary conditions""")
	
	if diff.shape[:2]!=(dim,dim):
		raise ValueError("OperatorMatrix error : inconsistent matrix dimensions")
		
	# -------- Decompose the tensors --------
	coef,offset = Selling.Decomposition(diff)
	nCoef = coef.shape[0]
	
	# ------ Check bounds or apply periodic boundary conditions -------
	grid = np.mgrid[tuple(slice(0,n) for n in bounds)]
	bGrid = np.broadcast_to(np.reshape(grid,(dim,1,)+bounds), offset.shape)

	neighPos = bGrid + offset
	neighNeg = bGrid - offset
	neumannPos = np.full(coef.shape,False)
	neumannNeg = np.full(coef.shape,False)
	dirichletPos = np.full(coef.shape,False)
	dirichletNeg = np.full(coef.shape,False)
	
	for neigh_,neumann,dirichlet in zip( (neighPos,neighNeg), (neumannPos,neumannNeg), (dirichletPos,dirichletNeg) ): 
		for neigh,cond_,bound in zip(neigh_,boundaryConditions,bounds): # Component by component
			for out,cond in zip( (neigh<0,neigh>=bound), cond_):
				if cond=='Periodic':
   				 	neigh[out] %= bound 
				elif cond=='Neumann':
					neumann[out] = True
				elif cond=='Dirichlet':
					dirichlet[out] = True
	
	# ------- Get the neighbor indices --------
	# Cumulative product in reverse order, omitting last term, beginning with 1
	cum = tuple(list(reversed(np.cumprod(list(reversed(bounds+(1,)))))))[1:]
	bCum = np.broadcast_to( np.reshape(cum, (dim,)+(1,)*(dim+1)), offset.shape)
	
	index = (bGrid*bCum).sum(0)
	indexPos = (neighPos*bCum).sum(0)
	indexNeg = (neighNeg*bCum).sum(0)
	
	# ------- Get the coefficients for the first order term -----
	if not omega is None:
		if intrinsicDrift:
			eta=omega
		else:
			eta = LP.dot_AV(LP.inverse(diff),omega)
			
		scalEta = LP.dot_VV(offset.astype(float), 
			np.broadcast_to(np.reshape(eta,(dim,1,)+bounds),offset.shape)) 
		coefOmega = coef*scalEta

	# ------- Create the triplets ------
	
	# Second order part
	# Nemann : remove all differences which are not inside (a.k.a multiply coef by inside)
	# TODO : Dirichlet : set to zero the coef only for the outside part
	
	coef = coef.flatten()/ (gridScale**2) # Take grid scale into account

	index = index.flatten()
	indexPos = indexPos.flatten()
	indexNeg = indexNeg.flatten()

	nff = lambda t : np.logical_not(t).astype(float).flatten()
	IP = nff(np.logical_or(neumannPos,dirichletPos))
	IN = nff(np.logical_or(neumannNeg,dirichletNeg))
	iP = nff(neumannPos)
	iN = nff(neumannNeg)

	if divergenceForm:
		row = np.concatenate((index, indexPos, index, indexPos))
		col = np.concatenate((index, index, indexPos, indexPos))
		data = np.concatenate((iP*coef/2, -IP*coef/2, -IP*coef/2, IP*coef/2))
		
		row  = np.concatenate(( row, index, indexNeg, index, indexNeg))
		col  = np.concatenate(( col, index, index, indexNeg, indexNeg))
		data = np.concatenate((data, iN*coef/2, -IN*coef/2, -IN*coef/2, IN*coef/2))
		
	else:
		row = np.concatenate( (index, index,	index))
		col = np.concatenate( (index, indexPos, indexNeg))
		data = np.concatenate((iP*coef+iN*coef, -IP*coef, -IN*coef))
	

	# First order part, using centered finite differences
	if not omega is None:	   
		coefOmega = coefOmega.flatten() / gridScale # Take grid scale in
		row = np.concatenate((row, index,	index))
		col = np.concatenate((col, indexPos, indexNeg))
		data= np.concatenate((data,IP*coefOmega/2,-IN*coefOmega/2))
	
	if not mult is None:
		# TODO Non periodic boundary conditions
		size=np.prod(bounds)
		row = np.concatenate((row, range(size)))
		col = np.concatenate((col, range(size)))
		data= np.concatenate((data,mult.flatten()))

	nz = data!=0
		
	return data[nz],(row[nz],col[nz])
	
	
	
	