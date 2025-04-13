# Metodo para resolver EDOs de 1a ordem Lineares com Runge-Kutta de 4a ordem e Euler

import numpy as np
import matplotlib.pyplot as plt

# Definindo a funcao f(x, y)
def f(x, y):
    return np.exp(-x) - 2 * y

# Solucao analitica
def y_analitica(x):
    return np.exp(-x)

# Condicoes iniciais e parametros
x0 = 0
y0 = 1
x_final = 2
h = 0.1
n = int((x_final - x0) / h) + 1

# Malha
x = np.linspace(x0, x_final, n)

# Vetores para armazenar as solucoes
y_euler = np.zeros(n)
y_rk4 = np.zeros(n)
y_exact = y_analitica(x)

# Condicao inicial
y_euler[0] = y0
y_rk4[0] = y0

# Metodo de Euler
for i in range(n - 1):
    y_euler[i + 1] = y_euler[i] + h * f(x[i], y_euler[i])

# Metodo de Runge-Kutta 4a ordem
for i in range(n - 1):
    k1 = f(x[i], y_rk4[i])
    k2 = f(x[i] + h / 2, y_rk4[i] + h * k1 / 2)
    k3 = f(x[i] + h / 2, y_rk4[i] + h * k2 / 2)
    k4 = f(x[i] + h, y_rk4[i] + h * k3)
    y_rk4[i + 1] = y_rk4[i] + (h / 6)*(k1 + 2*k2 + 2*k3 + k4)

# Plotando os graficos

plt.figure(figsize=(10, 6))
plt.plot(x, y_exact, label='Solucao analitica e^(-x)', color='red')
plt.plot(x, y_euler, label='Euler', color='Blue', linestyle='--')
plt.plot(x, y_rk4, label='RK4', color='green' , linestyle='dotted')
plt.plot(x, abs(y_euler - y_exact), label='Erro Euler', color='brown',linestyle='--', alpha=0.5)
plt.plot(x, abs(y_rk4 - y_exact), label='Erro RK4', color='black', linestyle='dotted', alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparacao: Euler vs RK4 vs Solucao Analitica')
plt.grid(True)
plt.legend()
plt.xticks(np.round(np.arange(0, x_final+0.1, 0.1), 1))
plt.show()
