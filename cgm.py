import numdifftools as nd
import numpy as np
import torch
import sys

#
#
#       запуск функции через команду "python .\cgm.py 0.01 1 14", где первое значение это точность, второе = х, третье = у в начале
#
#
def golden(f: callable, a: float, b: float, eps: float) -> list:          # Метод золотого сечения для одномерной минимизации.   
    GOLDEN_RATIO = ((5**0.5) - 1) / 2
    cal_val = 0

    while abs(a - b) > eps:
        d = GOLDEN_RATIO * (b - a)
        if f(a + d) > f(b - d):
            b = a + d
        else:
            a = b - d
        cal_val += 2

    return [(a + b) / 2, cal_val]



def f(var: list) -> float:                                             # задаём целевую функцию
    return (((var[0]-5)**2)*(var[1]-10)**2 +(var[0]-5)**2 + (var[1]-10)**2 + 1)

def f_t(x: float, y: float) -> float:                    #наша целевая функция для подсчёта матрицы гессе
    return (((x-5)**2)*(y-10)**2 +(x-5)**2 + (y-10)**2 + 1)



eps = float(sys.argv[1])                                                 # считываем параметры функции и точность 
start_p = np.array([[float(sys.argv[2])], [float(sys.argv[3])]])

f_val = []
rel_seq = []
iter: int = 0
cal_val: int = 0

rel_seq.append(start_p)                                  # начальная точка в последовательности 
iter += 1

f_val.append(f([start_p[0][0], start_p[1][0]]))               # значение целевой функции в начальной точке 

grad = nd.Gradient(f)                            # градиент целевой функции    

gradient = np.array([[0.], [0.]])                                                    # подсчёт градиента в начальной точке 
gradient[0][0], gradient[1][0] = grad([start_p[0][0], start_p[1][0]])
cal_val += 2

p = (-1)*gradient          # направление 


def is_update_moment(iterations: int) -> bool:                           #  "обновление" алгоритма для ослабления погрешностей 
    n = 2
    m = 1
    while m*n <= iterations:
        if m*n == iterations:
            return True
        m += 1
    return False


while ((gradient[0][0]**2) + (gradient[1][0]**2))**0.5 > eps:                       #  критерий остановки 
    current_point = rel_seq[iter - 1]

    def psi(xi):                                                     # нахождение х_i с помощью минимизации 
        x = current_point[0][0] + p[0][0]*abs(xi)
        y = current_point[1][0] + p[1][0]*abs(xi)
        return f([x, y])
  
    golden_eps = 0.0001
    golden_data = golden(f=psi, a=0., b=0.02, eps = golden_eps)
    xi = abs(golden_data[0])
    cal_val += golden_data[1]

    new_point = np.array([[0.], [0.]])                                    # нахождение новой точки в последовательности 
    new_point[0][0] = current_point[0][0] + p[0][0]*xi
    new_point[1][0] = current_point[1][0] + p[1][0]*xi                                                             
                                                                                
    gradient[0][0], gradient[1][0] = grad([new_point[0][0], new_point[1][0]])          #  нахождение градиента в новой точке 
    cal_val += 2

    if is_update_moment(iter):               # обновление алгоритма в случае необходимости 
        gamma = 0.
    else:
        hesse = torch.autograd.functional.hessian(f_t, (torch.Tensor([new_point[0][0]]), torch.Tensor([new_point[1][0]])))    # матрица гессе 
        cal_val += 4

        hesse = list(hesse)
        hesse = np.array([[hesse[0][0].item(), hesse[0][1].item()], [hesse[1][0].item(), hesse[1][1].item()]])
        gradient_mod = np.array([gradient[0][0], gradient[1][0]])
        p_mod = np.array([p[0][0], p[1][0]])

        gamma = (-1) * np.dot(np.matmul(hesse, p_mod), (-1)*gradient_mod) / np.dot(np.matmul(hesse, p_mod), p_mod)      # нахождение гаммы 
    
    p = gamma*p + (-1)*gradient              # нахождение вектора р

    rel_seq.append(new_point)                # вписываем точку в последовательность 
    iter += 1 

    f_val.append(f([new_point[0][0], new_point[1][0]]))       # целевая функция в новой точке     


def output_method_data(min_value: float, min_point: np.ndarray, iterations: int, calculated_values: int):    # выводим мин значение, точку, количество итераций и кол-во рассчитанных значений
    print(f"Минимальное значение: {min_value}")
    print(f"Точка: {min_point[0][0]}; {min_point[1][0]}")
    print(f"Итераций: {iterations - 1}")
    print(f"Вычесленных значений: {calculated_values}")

output_method_data(f_val[iter - 1], rel_seq[iter - 1], iter, cal_val)  # выводим данные 