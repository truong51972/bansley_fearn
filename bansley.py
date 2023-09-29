import numpy as np
import matplotlib.pyplot as plt
import random

def find_m(a, b, c, times):
    b1 = np.dot(a, b) + c
    for i in range(times - 1):
        b = np.dot(a, b) + c
    return round(b1[1][0]/b[1][0], 4)

def find_fp(detA, b, c, d):
    detB = abs(np.linalg.det(b))
    detC = abs(np.linalg.det(c))
    detD = abs(np.linalg.det(d))
    print(f'{detB} {detC} {detD}')
    Su=detA+detB+detC+detD
    u1=(detA)/Su
    u2=(detB)/Su
    u3=(detC)/Su
    u4=(detD)/Su

    fp1=u1
    fp2=u1+u2
    fp3=u1+u2+u3
    fp4=u1+u2+u3+u4
    return fp1, fp2, fp3, fp4
    
m1 = (1, 0, 0) 
m2 = (0, 0, 1)  
m3 = (0, 100/255, 0) 
m4 = (102/255, 0, 204/255) 
colors_list = [m1, m2, m3, m4]

E = [0.76, 0.07]
F = [-0.07, 0.76]
G = [0.24, -0.28]
H = [0.28, 0.24]
I = [-0.24, -0.28]
J = [-0.28, 0.24]

b = [[E[0], F[0]], [E[1], F[1]]]
c = [[G[0], H[0]], [G[1], H[1]]]
d = [[I[0], J[0]], [I[1], J[1]]]
e = [[0], [0]]
f = [[0], [1.5]]
g = [[0], [1.5]]
h = [[0], [0.5]]

m = find_m(b, e, f, 99)
a = [[0, 0], [0, m]]

color=[]
matrix_list = [a, b, c, d]
vector_list = [e, f, g, h]
N = []

fp1, fp2, fp3, fp4 = find_fp(0.0198, b, c, d)

I = np.round(np.random.rand(300000)*3).astype('int')
X=len(I)*[0] 
Y=len(I)*[0]

for i in range(len(I)): 
    N.append(random.random())
for i in range(0,len(N)):
    if N[i] < fp1:
        N[i] = 0
    elif N[i] < fp2:
        N[i] = 1
    elif N[i] < fp3:
        N[i] = 2
    elif N[i] <= fp4:
        N[i] = 3

for k in range(len(N)):
    previous_X_Y = [[X[k-1]], [Y[k-1]]]
    answer = np.dot(matrix_list[N[k]], previous_X_Y) + vector_list[N[k]]
    X[k], Y[k] = answer[0][0], answer[1][0]
    color.append(colors_list[N[k]])

plt.figure(figsize = (7,10))
plt.scatter(X, Y, 0.15, c=color[0:])
plt.show()