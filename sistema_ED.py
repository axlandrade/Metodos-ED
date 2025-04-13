# Metodo para resolver um sistema de EDOs de 1a ordem com Runge-Kutta de 4a ordem e Euler

import numpy as np
import matplotlib.pyplot as plt

# Parametros
h = 0.01
t0, t_end = 0, 2
n = int((t_end - t0) / h)

# Vetor tempo
t = np.linspace(t0, t_end, n+1)

# Inicializacao
x_euler = np.zeros(n+1)
y_euler = np.zeros(n+1)

x_rk4 = np.zeros(n+1)
y_rk4 = np.zeros(n+1)

# Condicoes iniciais
x_euler[0] = x_rk4[0] = 1
y_euler[0] = y_rk4[0] = 0

# Sistema de equacoes
def dx_dt(x, y):
    return 3*x + 4*y

def dy_dt(x, y):
    return -4*x + 3*y

# Euler
for i in range(n):
    x_euler[i+1] = x_euler[i] + h * dx_dt(x_euler[i], y_euler[i])
    y_euler[i+1] = y_euler[i] + h * dy_dt(x_euler[i], y_euler[i])

# RK4
for i in range(n):
    k1_x = h * dx_dt(x_rk4[i], y_rk4[i])
    k1_y = h * dy_dt(x_rk4[i], y_rk4[i])
    
    k2_x = h * dx_dt(x_rk4[i] + 0.5*k1_x, y_rk4[i] + 0.5*k1_y)
    k2_y = h * dy_dt(x_rk4[i] + 0.5*k1_x, y_rk4[i] + 0.5*k1_y)
    
    k3_x = h * dx_dt(x_rk4[i] + 0.5*k2_x, y_rk4[i] + 0.5*k2_y)
    k3_y = h * dy_dt(x_rk4[i] + 0.5*k2_x, y_rk4[i] + 0.5*k2_y)
    
    k4_x = h * dx_dt(x_rk4[i] + k3_x, y_rk4[i] + k3_y)
    k4_y = h * dy_dt(x_rk4[i] + k3_x, y_rk4[i] + k3_y)
    
    x_rk4[i+1] = x_rk4[i] + (k1_x + 2*k2_x + 2*k3_x + k4_x)/6
    y_rk4[i+1] = y_rk4[i] + (k1_y + 2*k2_y + 2*k3_y + k4_y)/6

# Solucao analitica
x_analitica = np.exp(3*t) * np.cos(4*t)
y_analitica = np.exp(3*t) * np.sin(4*t)

# Graficos
plt.figure(figsize=(18,10))

# x(t)
plt.subplot(2,2,1)
plt.plot(t, x_euler, '--', label='Euler x(t)', color='red')
plt.plot(t, x_rk4, '-', label='RK4 x(t)', color='blue')
plt.plot(t, x_analitica, '-', label='Analitica x(t)', color='green')
plt.title('Componente x(t)')
plt.xlabel('t')
plt.ylabel('x')
plt.grid(True)
plt.legend()

# y(t)
plt.subplot(2,2,2)
plt.plot(t, y_euler, '--', label='Euler y(t)', color='orange')
plt.plot(t, y_rk4, '-', label='RK4 y(t)', color='purple')
plt.plot(t, y_analitica, '-', label='Analitica y(t)', color='green')
plt.title('Componente y(t)')
plt.xlabel('t')
plt.ylabel('y')
plt.grid(True)
plt.legend()

# Erro em x(t)
plt.subplot(2,2,3)
plt.plot(t, abs(x_euler - x_analitica), '--', label='Erro Euler (x)', color='brown', alpha=0.7)
plt.plot(t, abs(x_rk4 - x_analitica), '-', label='Erro RK4 (x)', color='black', alpha=0.7)
plt.title('Erro absoluto em x(t)')
plt.xlabel('t')
plt.ylabel('|x_num - x_analitico|')
plt.grid(True)
plt.legend()

# Erro em y(t)
plt.subplot(2,2,4)
plt.plot(t, abs(y_euler - y_analitica), '--', label='Erro Euler (y)', color='brown', alpha=0.7)
plt.plot(t, abs(y_rk4 - y_analitica), '-', label='Erro RK4 (y)', color='black', alpha=0.7)
plt.title('Erro absoluto em y(t)')
plt.xlabel('t')
plt.ylabel('|y_num - y_analitico|')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.suptitle("Comparacao completa: Solucoes e Erros para o Sistema de EDOs", fontsize=15, y=1.02)
plt.show()
