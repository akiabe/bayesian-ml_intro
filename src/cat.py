import numpy as np
import math
from scipy import stats
from matplotlib import pyplot as plt

### likelihood function ###
K = 3
likelihood_pi = np.array([0.3, 0.5, 0.2])
s_k = np.identity(K)
prob_s_over_pi = stats.multinomial.pmf(x=s_k, n=1, p=likelihood_pi)
print(likelihood_pi)
print(s_k)
print(prob_s_over_pi)

k = np.arange(1, K+1)
#plt.bar(x=k, height=prob_s_over_pi)
#plt.xlabel("k")
#plt.ylabel("prob")
#plt.xticks(ticks=k, labels=k)
#plt.title("Categorical Distribution")
#plt.show()

### observation data ###
N = 50
X = np.random.multinomial(n=1, pvals=likelihood_pi, size=N)
sum_of_X = np.sum(X, axis=0)
print(f"X: {X[:5]}")
print(f"sum of X: {sum_of_X}")

#plt.bar(x=k, height=sum_of_X)
#plt.xlabel("k")
#plt.ylabel("count")
#plt.xticks(ticks=k, labels=k)
#plt.title("Observation Data")
#plt.show()

### prior distribution ###
alpha = np.array([1.0, 1.0, 1.0])

point_vec = np.arange(0.0, 1.001, 0.02)
x, y, z = np.meshgrid(point_vec, point_vec, point_vec)
pi = np.array([
    list(x.flatten()),
    list(y.flatten()),
    list(z.flatten())
]).T
pi = pi[1:, :]
pi /= np.sum(pi, axis=1, keepdims=True)
pi = np.unique(pi, axis=0)

prior = np.array([
    stats.dirichlet.pdf(x=pi[i], alpha=alpha) for i in range(len(pi))
])
print(prior)

tri_x = pi[:, 1] + pi[:, 2] / 2
tri_y = np.sqrt(3) * pi[:, 2] / 2
plt.scatter(tri_x, tri_y, cmap="jet")
plt.xlabel("pi_1, pi_2")
plt.ylabel("pi_1, pi_2")
plt.xticks(ticks=[0.0, 1.0], labels=["(1, 0, 0)", "(0, 1, 0)"])
plt.yticks(ticks=[0.0, 0.87], labels=["(1, 0, 0)", "(0, 0, 1)"])
plt.title("Dirichlet Distribution")
plt.colorbar()
#plt.gca().set_aspect('equal')
plt.show()

### posterior distribution ###
#a_hat = np.sum(X) + a
#b_hat = N - np.sum(X) + b
#print(a_hat)
#print(b_hat)

#posterior = stats.beta.pdf(x=prior_mu, a=a_hat, b=b_hat)
#print(posterior)

#plt.plot(prior_mu, posterior)
#plt.xlabel("mu")
#plt.ylabel("density")
#plt.title("Beta Distribution")
#plt.show()

### predict distribution ###
#pred_mu = a_hat / (a_hat + b_hat)
#print(pred_mu)

#pred = np.array([1-pred_mu, pred_mu])
#print(pred)

#plt.bar(x=x, height=pred)
#plt.xlabel("x")
#plt.ylabel("prob")
#plt.xticks(ticks=x, labels=x)
#plt.ylim(0.0, 1.0)
#plt.title("Bernoulli Distribution")
#plt.show()
