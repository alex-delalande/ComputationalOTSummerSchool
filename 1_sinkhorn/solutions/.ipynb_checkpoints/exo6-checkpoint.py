def sinkhorn_log(epsilon, niter):
    f = np.zeros(N[0])
    g = np.zeros(N[1])
    dual_obj = []
    errs = []
    for i in range(niter):
        # we stabilize the log-sum-exp operations by taking the maximum
        Mf = g.reshape(1,-1) - C
        Mf_max = np.max(Mf, axis = 1)
        f = - Mf_max - epsilon*np.log(np.sum( b.reshape(1,-1) * np.exp((Mf - Mf_max.reshape(-1,1))/epsilon), axis=1))
        
        marg = np.sum( a.reshape(-1,1) * np.exp((f.reshape(-1,1) + g.reshape(1,-1) - C)/epsilon), axis = 0)
        err = np.sum(b * np.abs(marg-1))
        
        Mg =  f.reshape(-1,1) - C
        Mg_max = np.max(Mg, axis = 0)
        g = - Mg_max- epsilon*np.log(np.sum( a.reshape(-1,1) * np.exp((Mg - Mg_max.reshape(1,-1))/epsilon), axis=0))
        
        val = sum(f * a) + sum(g * b) - epsilon * np.sum(a.reshape(-1,1)*b.reshape(1,-1)*np.exp((f.reshape(-1,1) + g.reshape(1,-1) - C)/epsilon))
        dual_obj = dual_obj + [val]
        errs = errs + [err]
    return f, g, dual_obj, errs


plt.figure(figsize = (10,7))

f, g, dual_obj1, errs1 = sinkhorn_log(0.1,100)
f, g, dual_obj2, errs2 = sinkhorn_log(0.01,500)
f, g, dual_obj3, errs3 = sinkhorn_log(0.005,750)

plt.subplot(121)
plt.semilogy(np.arange(100),np.asarray(dual_obj1[-1] - dual_obj1)+1e-8, linewidth = 2,label="epsilon=0.1")
plt.semilogy(np.arange(500),np.asarray(dual_obj2[-1] - dual_obj2)+1e-8, linewidth = 2,label="epsilon=0.01")
plt.semilogy(np.arange(750),np.asarray(dual_obj3[-1] - dual_obj3)+1e-8, linewidth = 2,label="epsilon=0.005")
plt.ylabel("Cvgce of dual objective")

plt.subplot(122)
plt.semilogy(np.arange(100),errs1,linewidth = 2,label="epsilon=0.1")
plt.semilogy(np.arange(500),errs2,linewidth = 2,label="epsilon=0.01")
plt.semilogy(np.arange(750),errs3,linewidth = 2,label="epsilon=0.005")
plt.ylabel("Cvgce of marginal constraint")

plt.legend()

plt.show()