import numpy as np
import math
from scipy import stats
from matplotlib import pyplot as plt

x = np.array([0, 1])
mu = 0.25
prob_x_over_mu = np.array([1-mu, mu])
print(prob_x_over_mu)

plt.bar(x=x, height=prob_x_over_mu)
plt