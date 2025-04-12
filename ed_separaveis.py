from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

def y_analitica(x):
    return 1 / (1 - 0.5 * x**2)

#Definição da EDO
def f(x,y):
    return x + y**2

#Método de Euler
def euler(f, y0, x):
    y=[y0]
    for i in range(1,len(x)):
        h=x[i]-x[i-1]
        y.append(y[-1] + h*f(x[i-1],y[-1]))
    return np.array(y)

#Método de Runge-Kutta de 4ª ordem

def rk4(f, y0, x):
    y = [y0]
    for i in range(1, len(x)):
        h = x[i] - x[i-1]
        xi = x[i-1]
        yi = y[-1]
        k1 = f(xi, yi)
        k2 = f(xi + h/2, yi + h/2 * k1)
        k3 = f(xi + h/2, yi + h/2 * k2)
        k4 = f(xi + h, yi + h * k3)
        y.append(yi + (h/6) * (k1 + 2*k2 + 2*k3 + k4)/6)
    return np.array(y)

#Malha de x com passo fixo

h = 0.1
x = np.arange(0, 1+h, h)
y0 = 1

#Solução

y_euler = euler(f, y0, x)

y_rk4 = rk4(f, y0, x)

y_exata = y_analitica(x)

#Plot

plt.plot(x, y_exata, label='Solução analítica $\dfrac{1}{1-\dfrac{x^2}{2}}$', linestyle='-', color='red')
plt.plot(x, y_euler, label='Euler', linestyle ='--', color='blue')
plt.plot(x, y_rk4, label='Runge-Kutta 4ª ordem', linestyle=':', color='green')
plt.plot(x, abs(y_euler - y_exata), label='Erro Euler', linestyle='--', color='brown', alpha=0.5)
plt.plot(x, abs(y_rk4 - y_exata), label='Erro RK4', linestyle=':', color='black', alpha=0.5)
plt.xticks(np.round(np.arange(0, 1.1, 0.1), 1))
plt.title('Comparação: Euler vs RK4 vs Solução Analítica')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
