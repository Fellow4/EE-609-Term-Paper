import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt
from random import randint

def prox(v, lmbda):
    return np.sign(v) * np.maximum(np.abs(v) - lmbda, 0.)

def error(A, S, A0, S0):
    val1 = np.linalg.norm(np.add(A, -A0), ord = 'fro', axis = None, keepdims = False)
    val2 = np.linalg.norm(np.add(S, -S0), ord = 'fro', axis = None, keepdims = False)
    val3 = np.linalg.norm(A0, ord = 'fro', axis = None, keepdims = False)
    val4 = np.linalg.norm(S0, ord = 'fro', axis = None, keepdims = False)
    return (val1/val3)

def objective(A, S, lmbda):
    val1 = np.linalg.norm(A, ord = 'nuc', axis = None, keepdims = False)
    val2 = np.count_nonzero(S)
    return val1 + lmbda*val2


#Generate the low rank component
n = 1000
cr = 0.05
cp = 0.05
r = (int)(cr * n)

U = np.random.normal(0, 1, (n, r))
V = np.random.normal(0, 1, (n, r))
L = np.matmul(U, V.transpose())

size = (int)(cp*n**2)

#Generate the support set Omega
N = (int)(size)
points = {(randint(0, n-1), randint(0, n-1)) for i in range(N)}
while len(points) < N:
        points |= {(randint(0, n-1), randint(0, n-1))}
points = list(list(x) for x in points)
points = np.array(points)

omega = np.array(points)
M_PI = math.pi
bound1 = math.sqrt(8*r/M_PI)
bound2 = 500

#generate the sparse component
S = np.zeros((n, n))

for i in range (size):
    x, y = (int)(omega[i][0]), (int)(omega[i][1])
    temp = np.random.uniform(low = -bound2, high = bound2, size = None)
    S[x][y] = temp

#declare the noise component for spca
snr = 50
rho = math.sqrt(((cp*8*r)/(3*M_PI) + cr*n)/pow(10, snr/10))
N = np.zeros((n, n))

for i in range(size):
    x, y = (int)(omega[i][0]), (int)(omega[i][1])
    temp = rho * np.random.normal(0, 1, size = None)
    N[x][y] = temp

#decalare the data matrix
data = np.add(S, L)
#data = np.add(data, N)
rank = np.linalg.matrix_rank(data, tol = None, hermitian = False)
u, s, v = np.linalg.svd(data, full_matrices = True)

spectral_norm = np.linalg.norm(data, ord = 2, axis = None, keepdims = False)
delta = pow(10, -5)
mu = 0.99 *  spectral_norm
mu_i = mu
lam = 1/(math.sqrt(n))

#RPCA via APGM
A0, A1 = np.zeros((n, n)), np.zeros((n, n))
S0, S1 = np.zeros((n, n)), np.zeros((n, n))
IA1 = np.array(A1)
IS1 = np.array(A1)
t0, t1 = 1, 1
MU = delta * mu
max_iter = 100

fista = []
x = []
ista = []
plt.xlabel('Iterations')
plt.ylabel('Error')

for iterations in range(max_iter):
    temp1 = np.add(A1, -A0)
    temp1 = ((t0-1)/t1) * temp1
    YA = np.add(A1, temp1)

    temp2 = np.add(S1, -S0)
    temp2 = ((t0-1)/t1) * temp1
    YS = np.add(S1, temp2)

    fista_err = error(A1, S1, L, S)
    ista_err = error(IA1, IS1, L, S)

    GA = np.add(YA, -0.5*np.add(np.add(YA, YS), -data))
    U, sigma, V = np.linalg.svd(GA, full_matrices = False)
    sigma = prox(np.diag(sigma), mu/2)
    A0 = np.array(A1)
    A1 = np.matmul(np.matmul(U, sigma), V)

    IGA = np.add(IA1, -0.5*np.add(np.add(IA1, IS1), -data))
    u, s, v = np.linalg.svd(IGA, full_matrices = False)
    s = prox(np.diag(s), mu/2)
    IA1 = np.matmul(np.matmul(u, s), v)


    GS = np.add(YS, -0.5*np.add(np.add(YA, YS), -data))
    S0 = np.array(S1)
    S1 = prox(S, (lam*mu)/2)

    IGS = np.add(IS1, -0.5*np.add(np.add(IA1, IS1), -data))
    IS1 = prox(IGS, (lam*mu)/2)

    temp = t0
    t0 = t1
    t1 = (1+math.sqrt(1+4*pow(temp, 2)))/2

    mu = max(0.9*mu, MU)
    x.append(iterations)
    fista.append(fista_err)
    ista.append(ista_err)


print("Error(ISTA) is : ", error(IA1, IS1, L, S))
print("Error(FISTA) is : ", error(A1, S1, L, S))
print("No of non-zero entries(ISTA) : ", np.count_nonzero(IS1))
print("No of non-zero entries(FISTA) : ", np.count_nonzero(S1))
min = objective(L, S, lam)
val1 = objective(IA1, IS1, lam)
val2 = objective(A1, S1, lam)
print("Minimum value is : ", min)
print("Estimated value(ISTA) is : ", val1)
print("Estimated value(FISTA) is : ", val2)

fista = np.array(fista)
x = np.array(x)
ista = np.array(ista)
plt.plot(x, fista, 'g^', label = 'FISTA')
plt.plot(x, ista, 'b-', label = 'ISTA')
plt.legend()
plt.show()
