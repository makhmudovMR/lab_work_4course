import math
import numpy as np
from numpy import linalg

def cubic_root(n):
    return n ** (1. / 3)


def processMatrixA(n):
    A = [[None] * n]*n
    for i in range(1, n+1):
        for j in range(1, n+1):
            if i == j:
                A[i-1][j-1] = n * (1+i) - i
            elif i != j:
                A[i-1][j-1] = math.sqrt(i*j) * (-1)**j
    return np.array(A)
                
def processMatrixB(n):
    _b = [None] * n
    for i in range(1, n+1):
        _b[i-1] = (1 + (-1)**i)/2
    return np.array(_b, ndmin=2).T


def process(n, n_inx, eps):

    b = processMatrixB(n[n_inx])
    print(b)
    A = processMatrixA(n[n_inx])
    print(A)


    #преобразуем Ax=b в x=Bx+c                             
    #-------------------------------------------------------

    # находим c
    c = []
    for indx in range(len(b)):
        c.append(b[indx] / A[indx][indx])
    c = np.array(c, ndmin = 2).T
    print(c)

    # находим B_2
    B_2 = np.zeros((n[n_inx],n[n_inx]))
    print(B_2)

    for i in range(len(B_2)):
        for j in range(len(B_2[i])):
            if i != j:
                B_2[i][j] = A[i][j] / A[i][i]

    # K
    K = round(math.log(eps*(1-linalg.norm(B_2))) / math.log(linalg.norm(B_2),math.e)) + 1
    print(K)
    # приближение 
    xi = []
    x = np.array([0,0,0,0], ndmin=2).T
    xi.append(x)
    for i in range(1, K):
        xi.append(None)
        xi[i] = np.dot(B_2,xi[i-1]) + c


    # Метод зейделя
    xz = []
    y = np.array([0,0,0,0])
    xz.append(y)
    for i in range(1, K):
        xz.append(np.array([None,None,None,None]))
        xz[i][0] = B_2[0][0] * xz[i-1][0] + B_2[0][1] * xz[i-1][1] + B_2[0][2] * xz[i-1][2] + B_2[0][3] * xz[i-1][3] + c[0]
        xz[i][1] = B_2[1][0] * xz[i][0] + B_2[1][1] * xz[i-1][1] + B_2[1][2] * xz[i-1][2] + B_2[1][3] * xz[i-1][3] + c[1]
        xz[i][2] = B_2[2][0] * xz[i][0] + B_2[2][1] * xz[i][1] + B_2[2][2] * xz[i-1][2] + B_2[2][3] * xz[i-1][3] + c[2]
        xz[i][3] = B_2[3][0] * xz[i][0] + B_2[3][1] * xz[i][1] + B_2[3][2] * xz[i][2] + B_2[3][3] * xz[i-1][3] + c[3]
        
    return xi, xz


def main():

    n = np.array([3,5,13])#Все n
    eps = 1.2 * (10**-4)#Точность вычисления

    result_iter, return_zeid = process(n, 1, eps)
    print(result_iter[-1])
    print(return_zeid[-1])
    str = """
    ---------------------------------------------------------------------------------------
    Решение методом итераций
    ---------------------------------------------------------------------------------------
    """
    

if __name__ == '__main__':
    main()
