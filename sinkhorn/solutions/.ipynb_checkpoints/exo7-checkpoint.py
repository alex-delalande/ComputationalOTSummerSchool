def sinkhorn_over(epsilon, niter,tau):
    f = np.zeros(N[0])
    g = np.zeros(N[1])
    dual_obj = []
    for i in range(niter):
        # we stabilize the log-sum-exp operations by taking the maximum
        Mf = g.reshape(1,-1) - C
        Mf_max = np.max(Mf, axis = 1)
        barf = - Mf_max - epsilon*np.log(np.sum( b.reshape(1,-1) * np.exp((Mf - Mf_max.reshape(-1,1))/epsilon), axis=1))
        f = (1+tau) * barf - tau * f 
        
        Mg =  f.reshape(-1,1) - C
        Mg_max = np.max(Mg, axis = 0)
        barg = - Mg_max- epsilon*np.log(np.sum( a.reshape(-1,1) * np.exp((Mg - Mg_max.reshape(1,-1))/epsilon), axis=0))
        g = (1+tau) * barg - tau * g 
        
        val = sum(f * a) + sum(g * b) - epsilon * np.sum(a.reshape(-1,1)*b.reshape(1,-1)*np.exp((f.reshape(-1,1) + g.reshape(1,-1) - C)/epsilon))
        dual_obj = dual_obj + [val]
    return f, g, dual_obj


plt.figure(figsize = (10,7))

f, g, dual_obj = sinkhorn_over(0.01,500,0.0)
plt.semilogy(np.arange(500),np.asarray(dual_obj[-1] - dual_obj)+1e-8, linewidth = 2,label="tau=0.0")
f, g, dual_obj = sinkhorn_over(0.01,500,0.5)
plt.semilogy(np.arange(500),np.asarray(dual_obj[-1] - dual_obj)+1e-8, linewidth = 2,label="tau=0.5")
f, g, dual_obj = sinkhorn_over(0.01,500,0.8)
plt.semilogy(np.arange(500),np.asarray(dual_obj[-1] - dual_obj)+1e-8, linewidth = 2,label="tau=0.8")
plt.ylabel("Cvgce of dual objective")

plt.title("Effect of the over-relaxation parameter")
plt.legend()

plt.show()