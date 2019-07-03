import numpy as np
import math

gamma = 500
epsilon = 1e-5

def k(x1,x2):
    return math.e**(-gamma*(np.linalg.norm(x1-x2)**2))

mat = np.loadtxt("data.txt")
X = mat[:,0:2]
y = mat[:,2]
n = X.shape[0]

w = np.ones((n))-0.95

K_tilda = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        K_tilda[i,j] = y[i] * y[j] * k(X[i], X[j])

def neg_dual_objective_func(w):
    ones = np.ones_like(w)
    return -(ones.dot(w)) + 0.5 * (w.T @ K_tilda @ w)

def grad_neg_dual_objective_func(w):
    ones = np.ones_like(w)
    temp = -ones + K_tilda @ w
    return temp


for step in range(300):
    w = w - 0.0001 * grad_neg_dual_objective_func(w)
    print(neg_dual_objective_func(w))