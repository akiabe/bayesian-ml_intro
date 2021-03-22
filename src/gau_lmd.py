import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

### likelihood function ###
mu = 25
lambda_truth = 0.01
x_line = np.arange(
    mu - 4 * np.sqrt(1 / lambda_truth),
    mu + 4 * np.sqrt(1 / lambda_truth),
    0.1
)

true_model = stats.norm.pdf(
    x=x_line,
    loc=mu,
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
    loc=mu,
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
a = 1
b = 1
lambda_line = np.arange(0, 4 * lambda_truth, 0.00001)

prior = stats.gamma.pdf(
    x=lambda_line,
    a=a,
    scale=1 / b,
)
print(prior[:10])

#plt.plot(lambda_line, prior)
#plt.xlabel("mu")
#plt.ylabel("density")
#plt.title("Gaussian Distribution")
#plt.show()

### posterior distribution ###
a_hat = 0.5 * N + a
b_hat = 0.5 * np.sum((x_n - mu)**2) + b
print(a_hat)
print(b_hat)

posterior = stats.gamma.pdf(
    x=lambda_line,
    a=a_hat,
    scale=1 / b_hat,
)

print(posterior[:10])

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
mu_s_hat = mu
lambda_s_hat = a_hat / b_hat
nu_s_hat = 2 * a_hat

predict = stats.t.pdf(
    x=x_line,
    df=nu_s_hat,
    loc=mu_s_hat,
    scale=np.sqrt(1 / lambda_s_hat),
)
print(predict[:10])

plt.plot(x_line, predict, label='predict')
plt.plot(x_line, true_model, label='true', linestyle='--')
plt.xlabel("x")
plt.ylabel("density")
plt.title("Gaussian Distribution")
plt.show()
