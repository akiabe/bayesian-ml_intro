import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

mu_truth = 0.25

x_point = np.array([0, 1])
true_model = np.array([1 - mu_truth, mu_truth])
true_model

plt.bar(x=x_point, height=true_model)
plt.xlabel("x")
plt.ylabel("prob")
plt.xticks(ticks=x_point, labels=x_point)
plt.ylim(0.0, 1.0)
plt.suptitle("Bernoulli Distribution")
plt.title(f"mu={mu_truth}", loc="left")
plt.show()