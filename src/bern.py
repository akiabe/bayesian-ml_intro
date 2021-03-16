import numpy as np
import math
from scipy import stats
from matplotlib import pyplot as plt

### likelihood function ###
x = np.array([0, 1])
mu = 0.25
prob_x_over_mu = np.array([1-mu, mu])
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
X = stats.bernoulli.rvs(p=mu, size=N)
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

### prior ###
