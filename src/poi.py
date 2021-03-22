import numpy as np
import math
from scipy import stats
from matplotlib import pyplot as plt

### likelihood function ###
lambda_truth = 4.0
x_line = np.arange(4 * lambda_truth)
true_model = stats.poisson.pmf(
    k=x_line,
    mu=lambda_truth,
)
print(np.round(true_model, 3))

#plt.bar(x=x_line, height=true_model)
#plt.xlabel("x")
#plt.ylabel("prob")
#plt.title("Poisson Distribution")
#plt.show()

### observation data ###
N = 50
x_n = np.random.poisson(lam=lambda_truth, size=N)
print(x_n[:10])

#plt.bar(x=x_line, height=[np.sum(x_n == x) for x in x_line])
#plt.xlabel("x")
#plt.ylabel("count")
#plt.title("Observation Data")
#plt.show()

### prior distribution ###
a = 1
b = 1
lambda_line = np.arange(0, 2 * lambda_truth, 0.001)
prior = stats.gamma.pdf(
    x=lambda_line,
    a=a,
    scale=1 / b,
)
print(np.round(prior, 3))

#plt.plot(lambda_line, prior)
#plt.xlabel("lambda")
#plt.ylabel("density")
#plt.title("Gamma Distribution")
#plt.show()

### posterior distribution ###
a_hat = np.sum(x_n) + a
b_hat = N + b
print(a_hat)
print(b_hat)

posterior = stats.gamma.pdf(
    x=lambda_line,
    a=a_hat,
    scale=1 / b_hat,
)
print(np.round(posterior, 5))

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
#plt.title("Gamma Distribution")
#plt.show()

### predict distribution ###
r_hat = a_hat
p_hat = 1 / (b_hat + 1)
print(r_hat)
print(p_hat)

predict = stats.nbinom.pmf(
    k=x_line,
    n=r_hat,
    p=1 - p_hat,
)
print(np.round(predict, 3))

plt.bar(
    x=x_line,
    height=true_model,
    label='true',
    alpha=0.5,
    color='white',
    edgecolor='red',
    linestyle='--',
)
plt.bar(
    x=x_line,
    height=predict,
    label='predict',
    alpha=0.5,
    color='purple',
)
plt.xlabel("x")
plt.ylabel("prob")
plt.title("Negative Binomial Distribution")
plt.show()
