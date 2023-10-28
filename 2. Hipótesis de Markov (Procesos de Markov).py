import numpy as np

# Matriz de transición de ejemplo para una Cadena de Markov
matriz_transicion = np.array([[0.7, 0.3],
                              [0.4, 0.6]])

# Estado inicial
estado_actual = 0  # Supongamos que comenzamos en el estado 0

# Realizar transiciones en la Cadena de Markov
num_transiciones = 100
historial_estados = [estado_actual]

for _ in range(num_transiciones):
    # Realizar una transición basada en la matriz de transición
    estado_actual = np.random.choice([0, 1], p=matriz_transicion[estado_actual])
    historial_estados.append(estado_actual)

print("Historial de estados:")
print(historial_estados)
