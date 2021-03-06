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
s_nk = np.random.multinomial(n=1, pvals=likelihood_pi, size=N)
sum_of_s_nk = np.sum(s_nk, axis=0)
print(f"X: {s_nk[:5]}")
print(f"sum of X: {sum_of_s_nk}")

#plt.bar(x=k, height=sum_of_X)
#plt.xlabel("k")
#plt.ylabel("count")
#plt.xticks(ticks=k, labels=k)
#plt.title("Observation Data")
#plt.show()

### prior distribution ###
alpha_k = np.array([1.0, 1.0, 1.0])

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
    stats.dirichlet.pdf(x=pi[i], alpha=alpha_k) for i in range(len(pi))
])
print(prior)

#tri_x = pi[:, 1] + pi[:, 2] / 2
#tri_y = np.sqrt(3) * pi[:, 2] / 2
#plt.scatter(tri_x, tri_y, cmap="jet")
#plt.xlabel("pi_1, pi_2")
#plt.ylabel("pi_1, pi_2")
#plt.xticks(ticks=[0.0, 1.0], labels=["(1, 0, 0)", "(0, 1, 0)"])
#plt.yticks(ticks=[0.0, 0.87], labels=["(1, 0, 0)", "(0, 0, 1)"])
#plt.title("Dirichlet Distribution")
#plt.colorbar()
#plt.gca().set_aspect('equal')
#plt.show()

### posterior distribution ###
alpha_hat_k = np.sum(s_nk, axis=0) + alpha_k
print(alpha_hat_k)

posterior = np.array([
    stats.dirichlet.pdf(x=pi[i], alpha=alpha_hat_k) for i in range(len(pi))
])
print(posterior)

### predict distribution ###
pi_hat_star_k = alpha_hat_k / np.sum(alpha_hat_k)
pi_hat_star_k = (np.sum(s_nk, axis=0) + alpha_k) / np.sum(np.sum(s_nk, axis=0) + alpha_k)
print(pi_hat_star_k)
