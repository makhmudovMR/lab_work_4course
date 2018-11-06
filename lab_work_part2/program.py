import math
from math import cos, sin, sqrt


def f(x):
    return math.sqrt(1-x) - math.cos(math.sqrt(1-x))


def f_d_1(x):
    return (sin(sqrt(1-x)) + 1) * (-1 / 2 * sqrt(1-x))


def f_d_2(x):
    return -1/4 * (sin(sqrt(1-x)) + 1) * (sqrt(1-x) ** -3) + 1/4 * cos(sqrt(1-x)) * sqrt((1-x)**2) ** -1



def half_divide_method2(a=0.0, b=1.0, f=f):
    e = 0.0001
    x = (a + b) / 2
    count = 0
    while math.fabs(f(x)) >= e:
        count += 1
        x = (a + b) / 2
        a, b = (a, x) if f(a) * f(x) < 0 else (x, b)
    return (a + b) / 2, count

def newtons_method(a=0.0, b=1.0, f=f, f1=f_d_1):
    e = 0.0001
    x0 = (a + b) / 2
    x1 = x0 - (f(x0) / f1(x0))
    count = 0
    while True:
        count += 1
        if math.fabs(x1 - x0) < e: return x1, count
        x0 = x1
        x1 = x0 - (f(x0) / f1(x0))


def simple_iteration_method(a=0.0, b=1.0, f=f):
    count = 0
    e = 10**-4
    i = 1
    flag = True
    arr = [0.5]
    while flag:
        count += 1
        x = f(arr[i-1])
        arr.append(x)
        if arr[i-1] - arr[i] < e:
            return arr[i], count
        i+=1

def main():
    h_d_m, h_d_m_c = half_divide_method2()
    n_m, n_m_c = newtons_method()
    s_i_m, s_i_m_c = simple_iteration_method()
    err_hdm = round(f(h_d_m))
    err_nm = round(f(n_m))
    err_sim = round(f(s_i_m))
    str  = """
    ------------------------------------------------------------------------------------
    Метод половинного деления: {0}   | Колл итераций: {1}      | {2} 
    ------------------------------------------------------------------------------------
    Метод Ньютона:             {3}   | Колл итераций: {4}      | {5} 
    ------------------------------------------------------------------------------------
    Метод простых итераций:    {6}   | Колл итераций: {7}      | {8} 
    ------------------------------------------------------------------------------------
    """.format(h_d_m, h_d_m_c, err_hdm, n_m, n_m_c, err_nm, s_i_m, s_i_m_c, err_sim)
    print(str)

if __name__ == '__main__':
    str = input("Лабораторная работа часть 2 (Магомед Махмудов): ")
    if str == "yes":
        main()