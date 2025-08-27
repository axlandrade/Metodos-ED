# ed_2ordem.py

import numpy as np
import matplotlib.pyplot as plt

# --- Definições do Sistema de EDOs ---
# A EDO y'' - 3y' + 2y = 0 é convertida em um sistema:
# y1' = y2
# y2' = -2*y1 + 3*y2
# onde y1 = y e y2 = y'

def sistema_derivadas(y1, y2):
    """
    Retorna as derivadas do sistema.
    
    Args:
        y1 (float): Valor atual de y.
        y2 (float): Valor atual de y'.
        
    Returns:
        tuple: (dy1/dt, dy2/dt)
    """
    dy1_dt = y2
    dy2_dt = -2*y1 + 3*y2
    return dy1_dt, dy2_dt

def solucao_analitica(x):
    """ Solução analítica da EDO para y(0)=1, y'(0)=0 """
    return 2 * np.exp(x) - np.exp(2 * x)

# --- Configuração da Simulação ---
h = 0.1                  # Passo
x0, x_end = 0, 2         # Intervalo
n = int((x_end - x0) / h)  # Número de passos

# Condições iniciais
y1_0 = 1  # y(0)
y2_0 = 0  # y'(0)

# Arrays
x = np.linspace(x0, x_end, n + 1)

# --- Solução Numérica com Euler ---
y1_euler, y2_euler = np.zeros(n + 1), np.zeros(n + 1)
y1_euler[0], y2_euler[0] = y1_0, y2_0

for i in range(n):
    dy1, dy2 = sistema_derivadas(y1_euler[i], y2_euler[i])
    y1_euler[i+1] = y1_euler[i] + h * dy1
    y2_euler[i+1] = y2_euler[i] + h * dy2

# --- Solução Numérica com Runge-Kutta de 4ª Ordem ---
y1_rk4, y2_rk4 = np.zeros(n + 1), np.zeros(n + 1)
y1_rk4[0], y2_rk4[0] = y1_0, y2_0

for i in range(n):
    # k1
    k1_y1, k1_y2 = sistema_derivadas(y1_rk4[i], y2_rk4[i])
    # k2
    k2_y1, k2_y2 = sistema_derivadas(y1_rk4[i] + 0.5*h*k1_y1, y2_rk4[i] + 0.5*h*k1_y2)
    # k3
    k3_y1, k3_y2 = sistema_derivadas(y1_rk4[i] + 0.5*h*k2_y1, y2_rk4[i] + 0.5*h*k2_y2)
    # k4
    k4_y1, k4_y2 = sistema_derivadas(y1_rk4[i] + h*k3_y1, y2_rk4[i] + h*k3_y2)
    
    y1_rk4[i+1] = y1_rk4[i] + (h/6) * (k1_y1 + 2*k2_y1 + 2*k3_y1 + k4_y1)
    y2_rk4[i+1] = y2_rk4[i] + (h/6) * (k1_y2 + 2*k2_y2 + 2*k3_y2 + k4_y2)

# --- Geração de Gráficos ---
y_analitica_vals = solucao_analitica(x)
erro_euler = np.abs(y1_euler - y_analitica_vals)
erro_rk4 = np.abs(y1_rk4 - y_analitica_vals)

plt.figure(figsize=(12, 8))
plt.plot(x, y_analitica_vals, 'g-', label="Solução Analítica: $2e^x - e^{2x}$", linewidth=2)
plt.plot(x, y1_euler, 'r--o', label="Euler", markersize=4)
plt.plot(x, y1_rk4, 'b--s', label="RK4", markersize=4)
plt.plot(x, erro_euler, '--', color='brown', label='Erro Euler', alpha=0.6)
plt.plot(x, erro_rk4, ':', color='black', label='Erro RK4', alpha=0.6)

plt.title("Solução Numérica para $y'' - 3y' + 2y = 0$")
plt.xlabel("x")
plt.ylabel("y / Erro")
plt.grid(True)
plt.legend()
plt.xlim([x0, x_end])
plt.ylim(bottom=-3)
plt.show()