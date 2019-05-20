def sinkhorn_annealing(epsilon_end, niter, epsilon_init, epsilon_fact):
    f = np.zeros(N[0])
    g = np.zeros(N[1])
    dual_obj = []
    epsilon = epsilon_init
    for i in range(niter):
        # we stabilize the log-sum-exp operations by taking the maximum
        Mf = g.reshape(1,-1) - C
        Mf_max = np.max(Mf, axis = 1)
        f = - Mf_max - epsilon*np.log(np.sum(b.reshape(1,-1) * np.exp((Mf - Mf_max.reshape(-1,1))/epsilon), axis=1))
        
        Mg =  f.reshape(-1,1) - C
        Mg_max = np.max(Mg, axis = 0)
        g = - Mg_max- epsilon*np.log(np.sum(a.reshape(-1,1) * np.exp((Mg - Mg_max.reshape(1,-1))/epsilon), axis=0))
        
        val = sum(f * a) + sum(g * b) - epsilon_end * np.sum(a.reshape(-1,1)*b.reshape(1,-1)*np.exp((f.reshape(-1,1) + g.reshape(1,-1) - C)/epsilon_end))
        dual_obj = dual_obj + [val]
        epsilon = max(epsilon_end,epsilon_fact*epsilon)
    return f, g, dual_obj

plt.figure(figsize = (10,7))
niter=200

f, g, dual_obj = sinkhorn_annealing(0.05,niter,0.05,0.0)
plt.semilogy(np.arange(niter),np.asarray(dual_obj[-1] - dual_obj)+1e-8, linewidth = 2,label="no annealing")
f, g, dual_obj = sinkhorn_annealing(0.05,niter,1.0,0.5)
plt.semilogy(np.arange(niter),np.asarray(dual_obj[-1] - dual_obj)+1e-8, linewidth = 2,label="beta=0.5")
f, g, dual_obj = sinkhorn_annealing(0.05,niter,1.0,0.8)
plt.semilogy(np.arange(niter),np.asarray(dual_obj[-1] - dual_obj)+1e-8, linewidth = 2,label="beta=0.8")

plt.title("cvgce of dual objective")
plt.legend()

plt.show()