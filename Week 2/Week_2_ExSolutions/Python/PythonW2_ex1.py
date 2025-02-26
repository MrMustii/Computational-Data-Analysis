# -*- coding: utf-8 -*-
"""
Computer intensive datahandling exercise 1 week 2

@author: dnor
"""
#%% Exercise a
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lng
import scipy.io
from sklearn import preprocessing

def ridgeMulti(X, _lambda, p, y):
    inner_prod = np.linalg.inv(np.matmul(X.T, X) + _lambda * np.eye(p,p))
    outer_prod = np.matmul(X.T, y)
    betas = np.matmul(inner_prod, outer_prod)
    return betas

#path = "C:\\Users\\dnor\\Desktop\\02582\\Lecture2\\S2"
prostatePath = 'Prostate.txt'

T = np.loadtxt(prostatePath, delimiter = ' ', skiprows = 1, usecols=[1,2,3,4,5,6,7,8,9])

y = T[:, 8]
X = T[:,:8]

[n, p] = np.shape(X)

k = 100; # try k values of lambda
lambdas = np.logspace(-4, 3, k)

betas = np.zeros((p,k))
    
for i in range(k):
    betas[:, i] = ridgeMulti(X, lambdas[i], p, y)
    
plt.figure()
plt.semilogx(lambdas, betas.T )
plt.xlabel("Lambdas")
plt.ylabel("Betas")
plt.title("Regularized beta estimates")

#%% Exercise b
K = 10
N = len(X)

I = np.asarray([0] * N)
for i in range(N):
    I[i] = (i + 1) % K + 1
     
I = I[np.random.permutation(N)]
lambdas = np.logspace(-4, 3, k)
MSE = np.zeros((10, 100))

for i in range(1, K+1):
    XTrain = X[i != I, :]
    yTrain = y[i != I]
    Xtest = X[i == I, :]
    yTest = y[i == I]

    # centralize and normalize data
    XTrain = preprocessing.scale(XTrain)
    yTrain = preprocessing.scale(XTrain)

    for j in range(100):
        Beta = ridgeMulti(XTrain, lambdas[j], 8, yTrain)
        MSE[(i - 1), j] = np.mean((yTest - np.matmul(Xtest, Beta)) ** 2)
        
meanMSE = np.mean(MSE, axis = 0)
jOpt = np.argsort(meanMSE)[0]

lambda_OP = lambdas[jOpt]

# Remember excact solution depends on a random indexing, so results may vary
plt.semilogx([lambda_OP, lambda_OP], [np.min(betas), np.max(betas)], marker = ".")

#%% Exercise c
seMSE = np.std(MSE, axis = 0) / np.sqrt(K)

J = np.where(meanMSE[jOpt] + seMSE[jOpt] > meanMSE)[0]
j = int(J[-1:])
Lambda_CV_1StdRule = lambdas[j]

print("CV lambda 1 std rule %0.2f" % Lambda_CV_1StdRule)

#%% Exercise 1 d
N = len(y)
[n, p] = np.shape(X)

off = np.ones(n)
M = np.c_[off, X] # Include offset / intercept

# Linear solver
beta, _, rnk, s = lng.lstsq(M, y)

yhat = np.matmul(M, beta)

e = y - np.matmul(X, Beta) # Low bias std
s = np.std(e)
D = np.zeros(100)
AIC = np.zeros(100)
BIC = np.zeros(100)

for j in range(100):
    Beta = ridgeMulti(XTrain, lambdas[j], 8, yTrain)
    inner = np.linalg.inv(np.matmul(X.T, X) + lambdas[j] *np.eye(8))
    outer = np.matmul(np.matmul(X, inner), X.T)
    D[j] = np.trace(outer)
    e = y - np.matmul(X, Beta)
    err = np.sum(e ** 2) / N
    AIC[j] = err + 2 * D[j] / N * s ** 2
    BIC[j] = N / (s ** 2) * (err + np.log(N) * D[j] / N * s ** 2)
    
jAIC = np.min(AIC)
jBIC = np.min(BIC)

print("AIC at %0.2f" % jAIC)
print("BIC at %0.2f" % jBIC)


#%% Exercise 1

NBoot = 100
[N, p] = np.shape(X)
Beta = np.zeros((p, len(lambdas), NBoot))

for i in range(NBoot):
    I = np.random.randint(0, N, N)
    XBoot = X[I, :]
    yBoot = y[I]
    for j in range(100):
        Beta[:, j, i] = ridgeMulti(XBoot, lambdas[j], p, yBoot)

stdBeta = np.std(Beta, axis = 2)
plt.figure()
for i in range(8):
    plt.semilogx(lambdas, stdBeta[i,:])
plt.title("Bootstrapped standard error")
plt.ylabel("Sigma of beta")
plt.xlabel("lambda")
