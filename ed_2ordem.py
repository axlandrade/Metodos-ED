#Método para Resolver ED de 2 ordem com Runge-Kutta de 4ª ordem e Euler

import numpy as np
import matplotlib.pyplot as plt

# Parâmetros da simulação
h = 0.1                  # passo
x0, x_end = 0, 2         # intervalo ajustado
n = int((x_end - x0)/h)  # número de passos

# Arrays para armazenar as soluções
x = np.linspace(x0, x_end, n+1)
y1_euler = np.zeros(n+1)  # y
y2_euler = np.zeros(n+1)  # y'

y1_rk4 = np.zeros(n+1)
y2_rk4 = np.zeros(n+1)

# Condições iniciais
y1_euler[0] = y1_rk4[0] = 1
y2_euler[0] = y2_rk4[0] = 0

# Funções derivadas
def f1(y1, y2):
    return y2

def f2(y1, y2):
    return 3*y2 - 2*y1

# Método de Euler
for i in range(n):
    y1_euler[i+1] = y1_euler[i] + h * f1(y1_euler[i], y2_euler[i])
    y2_euler[i+1] = y2_euler[i] + h * f2(y1_euler[i], y2_euler[i])

# Método de Runge-Kutta de 4ª ordem
for i in range(n):
    k1_y1 = h * f1(y1_rk4[i], y2_rk4[i])
    k1_y2 = h * f2(y1_rk4[i], y2_rk4[i])

    k2_y1 = h * f1(y1_rk4[i] + 0.5*k1_y1, y2_rk4[i] + 0.5*k1_y2)
    k2_y2 = h * f2(y1_rk4[i] + 0.5*k1_y1, y2_rk4[i] + 0.5*k1_y2)

    k3_y1 = h * f1(y1_rk4[i] + 0.5*k2_y1, y2_rk4[i] + 0.5*k2_y2)
    k3_y2 = h * f2(y1_rk4[i] + 0.5*k2_y1, y2_rk4[i] + 0.5*k2_y2)

    k4_y1 = h * f1(y1_rk4[i] + k3_y1, y2_rk4[i] + k3_y2)
    k4_y2 = h * f2(y1_rk4[i] + k3_y1, y2_rk4[i] + k3_y2)

    y1_rk4[i+1] = y1_rk4[i] + (k1_y1 + 2*k2_y1 + 2*k3_y1 + k4_y1)/6
    y2_rk4[i+1] = y2_rk4[i] + (k1_y2 + 2*k2_y2 + 2*k3_y2 + k4_y2)/6

# Solução analítica
y_analitica = 2*np.exp(x) - np.exp(2*x)

# Gráfico

plt.figure(figsize=(10, 6))
plt.plot(x, y_analitica, label="Solução Analítica", linestyle='-', color='green', linewidth=2)
plt.plot(x, y1_euler, label="Euler", linestyle='--', marker='o', color='red')
plt.plot(x, y1_rk4, label="RK4", linestyle='-', marker='s', color='blue')
plt.plot(x, abs(y1_euler - y_analitica), label='Erro Euler', linestyle='--', color='brown', alpha=0.5)
plt.plot(x, abs(y1_rk4 - y_analitica), label='Erro RK4', linestyle=':', color='black', alpha=0.5)
plt.title("Solução numérica e analítica da EDO $y'' - 3y' + 2y = 0$")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.xlim([0, 2])  # Limitar o eixo x até 2
plt.show()
