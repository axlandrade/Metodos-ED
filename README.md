# Métodos Numéricos para Equações Diferenciais Ordinárias (EDOs)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Libraries](https://img.shields.io/badge/Libraries-NumPy%20%7C%20Matplotlib-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Este repositório contém implementações em Python de métodos numéricos para a solução de Equações Diferenciais Ordinárias (EDOs). O foco principal é comparar a precisão e o desempenho do **Método de Euler** e do **Método de Runge-Kutta de 4ª Ordem (RK4)** com soluções analíticas conhecidas.

## 🎯 Objetivo

O projeto foi desenvolvido com fins educacionais para:
- Implementar algoritmos numéricos do zero.
- Visualizar e comparar a performance de diferentes métodos.
- Analisar o erro de aproximação de cada método.
- Fornecer uma base de código clara para estudantes e entusiastas de cálculo numérico.

## 📂 Estrutura do Repositório

O repositório está organizado com scripts dedicados a diferentes tipos de EDOs:

- **`ed_lineares_1ordem.py`**: Solução de EDOs lineares de 1ª ordem.
  - Exemplo: $y' = e^{-x} - 2y$

- **`ed_separaveis.py`**: Solução de EDOs separáveis de 1ª ordem.
  - Exemplo: $y' = x \cdot y^2$

- **`ed_2ordem.py`**: Solução de EDOs lineares de 2ª ordem, convertidas em um sistema de duas EDOs de 1ª ordem.
  - Exemplo: $y'' - 3y' + 2y = 0$

- **`sistema_ED.py`**: Solução de um sistema de EDOs de 1ª ordem.
  - Exemplo:
    - $x'(t) = 3x + 4y$
    - $y'(t) = -4x + 3y$

## 🛠️ Tecnologias Utilizadas

- **Python**: Linguagem de programação principal.
- **NumPy**: Para computação numérica e manipulação de arrays.
- **Matplotlib**: Para a geração de gráficos e visualização dos resultados.

## 🚀 Como Executar

### Pré-requisitos

Certifique-se de que você tem o Python 3 e as bibliotecas necessárias instaladas.

```bash
pip install numpy matplotlib