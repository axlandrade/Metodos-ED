# sistema_ED.py

import numpy as np
import matplotlib.pyplot as plt

# --- Definições do Sistema e Solução Analítica ---

def sistema_derivadas(x, y):
    """
    Define o sistema de EDOs:
    x'(t) = 3*x + 4*y
    y'(t) = -4*x + 3*y
    """
    dx_dt = 3*x + 4*y
    dy_dt = -4*x + 3*y
    return dx_dt, dy_dt

def solucao_analitica(t):
    """ Solução analítica para x(0)=1, y(0)=0 """
    x_t = np.exp(3*t) * np.cos(4*t)
    y_t = -np.exp(3*t) * np.sin(4*t) # Corrigido o sinal de acordo com a derivada dy_dt
    return x_t, y_t

# --- Configuração da Simulação ---
h = 0.01
t0, t_end = 0, 2
n = int((t_end - t0) / h)

# Vetor tempo
t = np.linspace(t0, t_end, n + 1)

# Condições iniciais
x0_val, y0_val = 1, 0

# --- Solução Numérica com Euler ---
x_euler, y_euler = np.zeros(n+1), np.zeros(n+1)
x_euler[0], y_euler[0] = x0_val, y0_val

for i in range(n):
    dx, dy = sistema_derivadas(x_euler[i], y_euler[i])
    x_euler[i+1] = x_euler[i] + h * dx
    y_euler[i+1] = y_euler[i] + h * dy

# --- Solução Numérica com Runge-Kutta de 4ª Ordem ---
x_rk4, y_rk4 = np.zeros(n+1), np.zeros(n+1)
x_rk4[0], y_rk4[0] = x0_val, y0_val

for i in range(n):
    # k1
    k1_x, k1_y = sistema_derivadas(x_rk4[i], y_rk4[i])
    # k2
    k2_x, k2_y = sistema_derivadas(x_rk4[i] + 0.5*h*k1_x, y_rk4[i] + 0.5*h*k1_y)
    # k3
    k3_x, k3_y = sistema_derivadas(x_rk4[i] + 0.5*h*k2_x, y_rk4[i] + 0.5*h*k2_y)
    # k4
    k4_x, k4_y = sistema_derivadas(x_rk4[i] + h*k3_x, y_rk4[i] + h*k3_y)
    
    x_rk4[i+1] = x_rk4[i] + (h/6) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
    y_rk4[i+1] = y_rk4[i] + (h/6) * (k1_y + 2*k2_y + 2*k3_y + k4_y)

# --- Geração de Gráficos ---
x_analitica, y_analitica_vals = solucao_analitica(t)

fig, axs = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
fig.suptitle("Comparação de Métodos Numéricos para Sistema de EDOs", fontsize=16)

# Gráfico para x(t)
axs[0, 0].plot(t, x_analitica, 'g-', label='Analítica x(t)', linewidth=2)
axs[0, 0].plot(t, x_euler, 'r--', label='Euler x(t)')
axs[0, 0].plot(t, x_rk4, 'b:', label='RK4 x(t)')
axs[0, 0].set_title('Componente x(t)')
axs[0, 0].set_xlabel('t')
axs[0, 0].set_ylabel('x(t)')
axs[0, 0].grid(True)
axs[0, 0].legend()

# Gráfico para y(t)
axs[0, 1].plot(t, y_analitica_vals, 'g-', label='Analítica y(t)', linewidth=2)
axs[0, 1].plot(t, y_euler, 'r--', label='Euler y(t)')
axs[0, 1].plot(t, y_rk4, 'b:', label='RK4 y(t)')
axs[0, 1].set_title('Componente y(t)')
axs[0, 1].set_xlabel('t')
axs[0, 1].set_ylabel('y(t)')
axs[0, 1].grid(True)
axs[0, 1].legend()

# Gráfico para Erro em x(t)
axs[1, 0].plot(t, np.abs(x_euler - x_analitica), 'r--', label='Erro Euler (x)')
axs[1, 0].plot(t, np.abs(x_rk4 - x_analitica), 'b-', label='Erro RK4 (x)')
axs[1, 0].set_title('Erro Absoluto em x(t)')
axs[1, 0].set_xlabel('t')
axs[1, 0].set_ylabel('Erro')
axs[1, 0].grid(True)
axs[1, 0].legend()
axs[1, 0].set_yscale('log') # Escala logarítmica para melhor visualização do erro

# Gráfico para Erro em y(t)
axs[1, 1].plot(t, np.abs(y_euler - y_analitica_vals), 'r--', label='Erro Euler (y)')
axs[1, 1].plot(t, np.abs(y_rk4 - y_analitica_vals), 'b-', label='Erro RK4 (y)')
axs[1, 1].set_title('Erro Absoluto em y(t)')
axs[1, 1].set_xlabel('t')
axs[1, 1].set_ylabel('Erro')
axs[1, 1].grid(True)
axs[1, 1].legend()
axs[1, 1].set_yscale('log') # Escala logarítmica

plt.show()