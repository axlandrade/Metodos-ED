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

#Malha de x com passo fixo

h = 0.1
x = np.arange(0, 1+h, h)
y0 = 1

#Solução

y_euler = euler(f, y0, x)

y_exata = y_analitica(x)

#Plot

plt.plot(x, y_euler, label='Euler', linestyle ='--', color='blue')
plt.plot(x, y_exata, label='Exata', linestyle='-', color='red')
plt.title('Comparação entre soluções numérica e exata')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
