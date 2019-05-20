epsilon = 0.1
niter = 1000

la = 2.0
r = la/(la+epsilon)
v = np.ones(N)
for i in range(niter):
    u = (a / (np.dot(K,v)))**r
    v = (b / (np.dot(np.transpose(K),u)))**r
P = np.dot(np.dot(np.diag(u),K),np.diag(v))


plt.figure(figsize=(13,7))

plt.subplot(231)
plt.imshow(np.log(P+1e-5))
#plt.axis('off');
plt.ylabel("lambda = 2.0")

plt.subplot(232)
plt.bar(t, np.dot(P,np.ones(N)), width = 1/len(t), color = "darkblue",label="marginal 1")
plt.legend()

plt.subplot(233)
plt.bar(t, np.dot(np.transpose(P),np.ones(N)), width = 1/len(t), color = "darkblue",label="marginal 2")
plt.legend()


la = 20.0
r = la/(la+epsilon)
v = np.ones(N)
for i in range(niter):
    u = (a / (np.dot(K,v)))**r
    v = (b / (np.dot(np.transpose(K),u)))**r
P = np.dot(np.dot(np.diag(u),K),np.diag(v))

plt.subplot(234)
plt.imshow(np.log(P+1e-5))
#plt.axis('off');
plt.ylabel("lambda = 20.0")

plt.subplot(235)
plt.bar(t, np.dot(P,np.ones(N)), width = 1/len(t), color = "darkblue",label="marginal 1")
plt.legend()

plt.subplot(236)
plt.bar(t, np.dot(np.transpose(P),np.ones(N)), width = 1/len(t), color = "darkblue",label="marginal 2")
plt.legend()
