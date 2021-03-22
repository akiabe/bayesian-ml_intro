import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

### likelihood function ###
mu_truth = 25
lmd = 0.01
x_line = np.linspace(
    mu_truth - 4 * np.sqrt(1 / lmd),
    mu_truth + 4 * np.sqrt(1 / lmd),
    num=1000
)
true_model = stats.norm.pdf(
    x=x_line,
    loc=mu_truth,
    scale=np.sqrt(1 / lmd),
)
print(true_model[:10])

#plt.plot(x_line, true_model)
#plt.xlabel("x")
#plt.ylabel("density")
#plt.title("Gaussian Distribution")
#plt.show()

### observation data ###
N = 50
x_n = np.random.normal(
    loc=mu_truth,
    scale=np.sqrt(1 / lmd),
    size=N,
)
print(x_n[:5])

#plt.hist(x=x_n, bins=50)
#plt.xlabel("x")
#plt.ylabel("count")
#plt.title("Gaussian Distribution")
#plt.show()

### prior distribution ###
m = 0
lambda_mu = 0.001
mu_line = np.linspace(
    mu_truth - 50,
    mu_truth + 50,
    num=1000,
)
prior = stats.norm.pdf(
    x=mu_line,
    loc=m,
    scale=np.sqrt(1 / lambda_mu),
)
print(prior[:10])

#plt.plot(mu_line, prior)
#plt.xlabel("mu")
#plt.ylabel("density")
#plt.title("Gaussian Distribution")
#plt.show()

### posterior distribution ###
lambda_mu_hat = N * lmd + lambda_mu
m_hat = (lmd * np.sum(x_n) + lambda_mu * m) / lambda_mu_hat
print(lambda_mu_hat)
print(m_hat)

posterior = stats.norm.pdf(
    x=mu_line,
    loc=m_hat,
    scale=np.sqrt(1 / lambda_mu_hat),
)
print(posterior[:10])

#plt.plot(mu_line, posterior)
#plt.vlines(
#    x=mu_truth,
#    ymin=0,
#    ymax=max(posterior),
#    color='red',
#    linestyle='--',
#)
#plt.xlabel("mu")
#plt.ylabel("density")
#plt.title("Gaussian Distribution")
#plt.show()

### predict distribution ###
lambda_star_hat = lmd * lambda_mu_hat / (lmd + lambda_mu_hat)
mu_star_hat = m_hat
print(lambda_star_hat)
print(mu_star_hat)

predict = stats.norm.pdf(
    x=x_line,
    loc=mu_star_hat,
    scale=np.sqrt(1 / lambda_star_hat),
)
print(predict[:10])

plt.plot(x_line, predict, label='predict')
plt.plot(x_line, true_model, label='true', linestyle='--')
plt.xlabel("x")
plt.ylabel("density")
plt.title("Gaussian Distribution")
plt.show()