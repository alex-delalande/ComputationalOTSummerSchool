def sinkhorn_iter(g, C, epsilon):
    "Perform one stable sinkhorn iteration, and evaluate the convergence"
    Mf = g.reshape(1,-1) - C
    Mf_max = np.max(Mf, axis = 1)
    f = - Mf_max - epsilon*np.log(np.sum( b.reshape(1,-1) * np.exp((Mf - Mf_max.reshape(-1,1))/epsilon), axis=1))
        
    marg = np.sum( a.reshape(-1,1) * np.exp((f.reshape(-1,1) + g.reshape(1,-1) - C)/epsilon), axis = 0)
    err = np.sum(np.abs(marg-1))
    
    Mg =  f.reshape(-1,1) - C
    Mg_max = np.max(Mg, axis = 0)
    gg = - Mg_max- epsilon*np.log(np.sum(a.reshape(-1,1) * np.exp((Mg - Mg_max.reshape(1,-1))/epsilon), axis=0))
    
    return gg, err

def sinkhorn_RNA(epsilon, niter, la, k):
    m   = np.shape(C)[1]
    gs  = np.zeros((m,k))
    Tgs = np.zeros((m,k))
    errs= []
    for i in range(k-1):
        # first compute at least k iterations to populate the matrix of residuals
        Tgs[:,i], err = sinkhorn_iter(gs[:,i], C, epsilon)
        gs[:,i+1]= np.copy(Tgs[:,i])
        errs = errs +[err]

    Tgs[:,k-1],err = sinkhorn_iter(gs[:,k-1], C, epsilon)
    errs = errs +[err]

    for i in range(niter):
        # now perform the accelerated algorithm
        R = Tgs - gs
        c = RNA_coeffs(R,la)
        g = np.sum(c * Tgs, axis=1)
        gs = np.hstack((gs[:,1:],g.reshape(-1,1)))
        Tg, err = sinkhorn_iter(g, C, epsilon)
        Tgs = np.hstack((Tgs[:,1:],Tg.reshape(-1,1)))
        errs = errs +[err]
    return errs

errs1 = sinkhorn_RNA(0.1, 50, 1e-10, 1)
errs2 = sinkhorn_RNA(0.1, 50, 1e-10, 7)

plt.figure(figsize = (10,7))
niter=200

plt.semilogy(errs1,label="No acceleration")
plt.semilogy(errs2,label="Acceleration, k=7")
plt.ylabel("|| P 1 - a ||_1")

plt.legend()

plt.show()