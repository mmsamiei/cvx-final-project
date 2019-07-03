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

def dual_objective_func(w):
    ones = np.ones_like(w)
    return ones.dot(w) - 0.5 * (w.T @ K_tilda @ w)

def barier_term(w):
    temp = 0
    for i in range(w.shape[0]):
        if(C - w[i] <= 0 ):
            return 99999999999
        temp = temp + math.log2(C - w[i])
        if(w[i] <= 0):
            return 99999999999
        temp = temp + math.log2(w[i])
    temp = -temp
    return temp

def barier_func(w, t):
    return t*dual_objective_func(w) + barier_term(w)

def grad_barier_func(w, t):
    ones = np.ones_like(w)
    temp = t * (ones - K_tilda @ w)
    for i in range(w.shape[0]):
        e = np.eye(1, w.shape[0], i)
        temp = temp + (1/(C-w[i]) - (1/w[i])) * e
    return temp

def hess_barier_func(w, t):
    temp = t * K_tilda
    for i in range(K_tilda.shape[0]):
        temp[i,i] = temp[i,i] + (1/(C-w[i])**2) + (1 / (w[i])**2)
    return temp

##Barrier algorithme
t = 10000
u = 4
C = 0.1
for step in range(30):
    #print(barier_func(w,t))
    grad = grad_barier_func(w, t).ravel()
    hess = hess_barier_func(w, t)
    ## now cal the delta direction
    left_matrix = np.zeros((w.shape[0]+1, w.shape[0]+1))
    for i in range(left_matrix.shape[0]-1):
        for j in range(left_matrix.shape[0]-1):
            left_matrix[i][j] = hess[i][j]
    for i in range(left_matrix.shape[0]-1):
        left_matrix[i][-1] = y[i]
    for i in range(left_matrix.shape[0] - 1):
        left_matrix[-1][i] = y[i]
    right_matrix = np.zeros(w.shape[0]+1)
    for i in range(right_matrix.shape[0]-1):
        right_matrix[i] = -grad[i]
    middle_matrix = np.linalg.inv(left_matrix) @ right_matrix
    delta_newton = middle_matrix[:-1]
    ###
    ### now lets backtrack
    alpha = 0.01
    beta = 0.5
    t_backtrack = 1
    while (barier_func(w + t_backtrack * delta_newton, t) > barier_func(w, t) + alpha * t_backtrack * (grad.T @ delta_newton)):
        t_backtrack = beta * t_backtrack
    ### now lets backtrack
    print("t back_track is: ",t_backtrack)
    w = w + t_backtrack * delta_newton
    print(dual_objective_func(w))
    t = t * u
