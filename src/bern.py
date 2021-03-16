import numpy as np
import math
from scipy import stats
from matplotlib import pyplot as plt

### likelihood function ###
x = np.array([0, 1])
likelihood_mu = 0.25
prob_x_over_mu = np.array([1-likelihood_mu, likelihood_mu])
print(prob_x_over_mu)

#plt.bar(x=x, height=prob_x_over_mu)
#plt.xlabel("x")
#plt.ylabel("prob")
#plt.xticks(ticks=x, labels=x)
#plt.ylim(0.0, 1.0)
#plt.title("Bernoulli Distribution")
#plt.show()

### observation data ###
N = 50
X = stats.bernoulli.rvs(p=likelihood_mu, size=N)
print(X)
num_of_1 = np.sum(X)
num_of_0 = N - num_of_1
print(f"num of 0: {num_of_0}")
print(f"num of 1: {num_of_1}")

#plt.bar(x=x, height=[num_of_0, num_of_1])
#plt.xlabel("x")
#plt.ylabel("count")
#plt.xticks(ticks=x, labels=x)
#plt.title("Observation Data")
#plt.show()

### prior distribution ###
a = 1.0
b = 1.0
prior_mu = np.arange(0.0, 1.001, 0.001)
prior = stats.beta.pdf(x=prior_mu, a=a, b=b)
print(prior)

#plt.plot(prior_mu, prior)
#plt.xlabel("mu")
#plt.ylabel("density")
#plt.xticks(ticks=x, labels=x)
#plt.title("Beta Distribution")
#plt.show()

### posterior distribution ###
a_hat = np.sum(X) + a
b_hat = N - np.sum(X) + b
print(a_hat)
print(b_hat)

posterior = stats.beta.pdf(x=prior_mu, a=a_hat, b=b_hat)
print(posterior)

#plt.plot(prior_mu, posterior)
#plt.xlabel("mu")
#plt.ylabel("density")
#plt.title("Beta Distribution")
#plt.show()

### predict distribution ###
pred_mu = a_hat / (a_hat + b_hat)
print(pred_mu)

pred = np.array([1-pred_mu, pred_mu])
print(pred)

plt.bar(x=x, height=pred)
plt.xlabel("x")
plt.ylabel("prob")
plt.xticks(ticks=x, labels=x)
plt.ylim(0.0, 1.0)
plt.title("Bernoulli Distribution")
plt.show()
