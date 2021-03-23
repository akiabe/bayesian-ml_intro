import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

### likelihood function ###
mu_truth = 25
lambda_truth = 0.01
x_line = np.linspace(
    mu_truth - 4 * np.sqrt(1 / lambda_truth),
    mu_truth + 4 * np.sqrt(1 / lambda_truth),
    num=1000
)

true_model = stats.norm.pdf(
    x=x_line,
    loc=mu_truth,
    scale=np.sqrt(1 / lambda_truth),
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
    scale=np.sqrt(1 / lambda_truth),
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
beta = 1
a = 1
b = 1
E_lambda = a / b

mu_line = np.linspace(
    mu_truth - 50,
    mu_truth + 50,
    num=1000,
)

prior_mu = stats.norm.pdf(
    x=mu_line,
    loc=m,
    scale=np.sqrt(1 / beta / E_lambda),
)
print(prior_mu[:10])

#plt.plot(mu_line, prior_mu)
#plt.xlabel("mu")
#plt.ylabel("density")
#plt.title("Gaussian Distribution")
#plt.show()

lambda_line = np.linspace(0, 4 * lambda_truth, num=1000)
prior_lambda = stats.gamma.pdf(
    x=lambda_line,
    a=a,
    scale=1 / b,
)
print(lambda_line[:10])
print(prior_lambda[:10])

plt.plot(lambda_line, prior_lambda)
plt.xlabel("lambda")
plt.ylabel("density")
plt.title("Gamma Distribution")
plt.show()

### posterior distribution ###
#a_hat = 0.5 * N + a
#b_hat = 0.5 * np.sum((x_n - mu)**2) + b
#print(a_hat)
#print(b_hat)

#posterior = stats.gamma.pdf(
#    x=lambda_line,
#    a=a_hat,
#    scale=1 / b_hat,
#)

#print(posterior[:10])

#plt.plot(lambda_line, posterior)
#plt.vlines(
#    x=lambda_truth,
#    ymin=0,
#    ymax=max(posterior),
#    color='red',
#    linestyle='--',
#)
#plt.xlabel("lambda")
#plt.ylabel("density")
#plt.title("Gaussian Distribution")
#plt.show()

### predict distribution ###
#mu_s_hat = mu
#lambda_s_hat = a_hat / b_hat
#nu_s_hat = 2 * a_hat

#predict = stats.t.pdf(
#    x=x_line,
#    df=nu_s_hat,
#    loc=mu_s_hat,
#    scale=np.sqrt(1 / lambda_s_hat),
#)
#print(predict[:10])

#plt.plot(x_line, predict, label='predict')
#plt.plot(x_line, true_model, label='true', linestyle='--')
#plt.xlabel("x")
#plt.ylabel("density")
#plt.title("Gaussian Distribution")
#plt.show()
