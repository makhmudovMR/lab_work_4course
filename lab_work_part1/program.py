import math
import numpy as np
from numpy import linalg

def cubic_root(n):
    return n ** (1. / 3)

def process1(B, n, n_inx, eps):

    b = np.array([math.sqrt(n[n_inx])-1, math.sqrt(n[n_inx])+1, cubic_root(n[n_inx])-1, cubic_root(n[n_inx])+1])
    K = 0
    A = linalg.inv(B) + B # находим матрицу A = inv(B)+B
    #преобразуем Ax=b в x=Bx+c

    # находим c
    c = []
    for indx in range(len(b)):
        c.append(b[indx] / A[indx][indx])
    c = np.array(c, ndmin=2).T

    # находим B_2
    B_2 = np.zeros((4,4))

    for i in range(len(B_2)):
        for j in range(len(B_2[i])):
            if i != j:
                B_2[i][j] = A[i][j] / A[i][i]

    # K
    K = round(math.log(eps*(1-linalg.norm(B_2))) / math.log(linalg.norm(B_2),math.e)) + 1
    # приближение 
    xi = []
    x = np.array([0,0,0,0], ndmin=2).T
    xi.append(x)
    count_iter = 0
    for i in range(1, K):
        xi.append(None)
        xi[i] = np.dot(B_2,xi[i-1]) + c
        count_iter = count_iter + 1
        if xi[i][0] - xi[i-1][0] < eps:
            break
                  


    # Метод зейделя
    xz = []
    y = np.array([0,0,0,0])
    xz.append(y)
    count_zeid = 0
    for i in range(1, K):
        xz.append(np.array([None,None,None,None]))
        xz[i][0] = B_2[0][0] * xz[i-1][0] + B_2[0][1] * xz[i-1][1] + B_2[0][2] * xz[i-1][2] + B_2[0][3] * xz[i-1][3] + c[0]
        xz[i][1] = B_2[1][0] * xz[i][0] + B_2[1][1] * xz[i-1][1] + B_2[1][2] * xz[i-1][2] + B_2[1][3] * xz[i-1][3] + c[1]
        xz[i][2] = B_2[2][0] * xz[i][0] + B_2[2][1] * xz[i][1] + B_2[2][2] * xz[i-1][2] + B_2[2][3] * xz[i-1][3] + c[2]
        xz[i][3] = B_2[3][0] * xz[i][0] + B_2[3][1] * xz[i][1] + B_2[3][2] * xz[i][2] + B_2[3][3] * xz[i-1][3] + c[3]
        
        count_zeid +=1
        if xi[i][0] - xi[i-1][0] < eps:
            break
    
    return xi, xz, count_iter, count_zeid




    # Метод зейделя
    xz = []
    y = np.array([0,0,0,0])
    xz.append(y)
    count_zeid = 0
    for i in range(1, K):
        xz.append(np.array([None,None,None,None]))
        xz[i][0] = B_2[0][0] * xz[i-1][0] + B_2[0][1] * xz[i-1][1] + B_2[0][2] * xz[i-1][2] + B_2[0][3] * xz[i-1][3] + c[0]
        xz[i][1] = B_2[1][0] * xz[i][0] + B_2[1][1] * xz[i-1][1] + B_2[1][2] * xz[i-1][2] + B_2[1][3] * xz[i-1][3] + c[1]
        xz[i][2] = B_2[2][0] * xz[i][0] + B_2[2][1] * xz[i][1] + B_2[2][2] * xz[i-1][2] + B_2[2][3] * xz[i-1][3] + c[2]
        xz[i][3] = B_2[3][0] * xz[i][0] + B_2[3][1] * xz[i][1] + B_2[3][2] * xz[i][2] + B_2[3][3] * xz[i-1][3] + c[3]
        
        count_zeid +=1
        if xi[i][0] - xi[i-1][0] < eps:
            break
    
    return xi, xz, count_iter, count_zeid

    


def main():
    B = np.array(
        [
            [3,0,2,1],
            [4,10,0,2],
            [1,0,5,1],
            [1,1,1,5],
        ]
    )
    n = np.array([1,4,5])
    eps = 0.6 * (10**-4)

    # result_iter, result_zeid, count_iter, count_zeid = process1(B, n, 1, eps)
    result_iter, result_zeid, count_iter, count_zeid = process1(B, n, 1, eps)
    iter = np.array([item for item in result_iter[-1]])
    zeid = np.array([item for item in result_zeid[-1]])


    abs_differnce = iter - zeid
    
    str = """
    -------------------------------------------------------------------------------------------------
    Решение методом итераций |Решение методом Зейделя | Модуль разности | Колличество итераций      |
    -------------------------------------------------------------------------------------------------
    x1 = {0}        |x1 = {4}    |  |X_p - X_eps| = {8:.6f}     | K_iter = {12}
    x2 = {1}        |x2 = {5}    |  |X_p - X_eps| = {9:.6f}     |
    x3 = {2}        |x3 = {6}    |  |X_p - X_eps| = {10:.6f}    |
    x4 = {3}        |x4 = {7}    |  |X_p - X_eps| = {11:.6f}    | K_zeid = {13}
    -------------------------------------------------------------------------------------------------
    """.format(iter[0], iter[1], iter[2], iter[3], \
    zeid[0],zeid[1], zeid[2], zeid[3], \
    abs_differnce[0][0],abs_differnce[1][0],abs_differnce[2][0],abs_differnce[3][0],\
    count_iter, count_zeid)
    print(str)

if __name__ == '__main__':
    main()