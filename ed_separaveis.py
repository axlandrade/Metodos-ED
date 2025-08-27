# ed_separaveis.py

import numpy as np
import matplotlib.pyplot as plt

# --- Definições da EDO e Solução Analítica ---

def f(x, y):
    """ Define a EDO: y' = x * y^2 """
    return x * y**2

def y_analitica(x):
    """ Solução analítica da EDO com y(0)=1: y(x) = 1 / (1 - 0.5*x^2) """
    # Adicionado tratamento para evitar divisão por zero na singularidade
    with np.errstate(divide='ignore', invalid='ignore'):
        return 1 / (1 - 0.5 * x**2)

# --- Métodos Numéricos (podem ser importados de um arquivo comum) ---

def euler(f, y0, x):
    """ Resolve y' = f(x, y) usando o método de Euler. """
    y = np.zeros_like(x)
    y[0] = y0
    for i in range(len(x) - 1):
        h = x[i+1] - x[i]
        y[i+1] = y[i] + h * f(x[i], y[i])
    return y

def rk4(f, y0, x):
    """ Resolve y' = f(x, y) usando o método de Runge-Kutta de 4ª Ordem. """
    y = np.zeros_like(x)
    y[0] = y0
    for i in range(len(x) - 1):
        h = x[i+1] - x[i]
        xi, yi = x[i], y[i]
        
        k1 = f(xi, yi)
        k2 = f(xi + h / 2, yi + h * k1 / 2)
        k3 = f(xi + h / 2, yi + h * k2 / 2)
        k4 = f(xi + h, yi + h * k3)
        
        y[i+1] = yi + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return y

# --- Configuração da Simulação ---

y0 = 1
x_inicial = 0
x_final = 1.2  # Intervalo ajustado para visualizar antes da singularidade em x=sqrt(2)
h = 0.1
x = np.arange(x_inicial, x_final + h, h)

# --- Execução e Geração do Gráfico ---

# Soluções
y_exata = y_analitica(x)
y_euler = euler(f, y0, x)
y_rk4 = rk4(f, y0, x)

# Erros
erro_euler = np.abs(y_euler - y_exata)
erro_rk4 = np.abs(y_rk4 - y_exata)

# Plot
plt.figure(figsize=(12, 8))
plt.plot(x, y_exata, 'r-', label='Solução Analítica: $1/(1-x^2/2)$', linewidth=2)
plt.plot(x, y_euler, 'b--o', label='Euler', markersize=4)
plt.plot(x, y_rk4, 'g--s', label='RK4', markersize=4)
plt.plot(x, erro_euler, '--', color='brown', label='Erro Euler', alpha=0.6)
plt.plot(x, erro_rk4, ':', color='black', label='Erro RK4', alpha=0.6)

plt.title("Comparação de Métodos para $y' = x \cdot y^2$")
plt.xlabel('x')
plt.ylabel('y / Erro')
plt.legend()
plt.grid(True)
plt.ylim(bottom=0)
plt.show()