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

plt.bar(x=k, height=sum_of_X)
plt.xlabel("k")
plt.ylabel("count")
plt.xticks(ticks=k, labels=k)
plt.title("Observation Data")
plt.show()

### prior distribution ###
alpha = np.array([1.0, 1.0, 1.0])
point_vec = np.arange(0.0, 1.001, 0.002)
x, y, z = np.meshgrid(point_vec, point_vec, point_vec)
pi = np.array([
    list(x.flatten()),
    list(y.flatten()),
    list(z.flatten())
])
print(pi)

#b = 1.0
#prior_mu = np.arange(0.0, 1.001, 0.001)
#prior = stats.beta.pdf(x=prior_mu, a=a, b=b)
#print(prior)

#plt.plot(prior_mu, prior)
#plt.xlabel("mu")
#plt.ylabel("density")
#plt.xticks(ticks=x, labels=x)
#plt.title("Beta Distribution")
#plt.show()

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
