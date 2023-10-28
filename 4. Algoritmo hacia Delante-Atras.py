import numpy as np

# Modelo oculto de Markov (HMM) de ejemplo
# Supongamos un HMM con 3 estados y 2 observaciones posibles
num_estados = 3
num_observaciones = 2

# Matriz de transición de estado
transicion_estado = np.array([[0.7, 0.2, 0.1],
                              [0.3, 0.5, 0.2],
                              [0.1, 0.3, 0.6]])

# Probabilidad inicial del estado
probabilidad_inicial = np.array([0.3, 0.4, 0.3])

# Secuencia observada
secuencia_observada = [0, 1, 0, 1, 1]  # Ejemplo de una secuencia de observaciones

# Algoritmo hacia adelante
def forward_algorithm(secuencia_observada):
    T = len(secuencia_observada)
    alpha = np.zeros((T, num_estados))

    # Paso hacia adelante
    for t in range(T):
        observacion = secuencia_observada[t]
        if t == 0:
            alpha[t, :] = probabilidad_inicial * transicion_estado[:, observacion]
        else:
            for j in range(num_estados):
                alpha[t, j] = np.sum(alpha[t - 1, :] * transicion_estado[:, j]) * transicion_estado[j, observacion]

    return alpha

# Algoritmo hacia atrás
def backward_algorithm(secuencia_observada):
    T = len(secuencia_observada)
    beta = np.zeros((T, num_estados))

    # Paso hacia atrás
    for t in range(T - 1, -1, -1):
        observacion = secuencia_observada[t]
        if t == T - 1:
            beta[t, :] = 1.0
        else:
            for i in range(num_estados):
                beta[t, i] = np.sum(transicion_estado[i, :] * transicion_estado[:, observacion] * beta[t + 1, :])

    return beta

# Estimación del estado oculto en cada paso de tiempo
alpha = forward_algorithm(secuencia_observada)
beta = backward_algorithm(secuencia_observada)
estado_oculto_estimado = np.argmax(alpha * beta, axis=1)

print("Secuencia observada:", secuencia_observada)
print("Estado oculto estimado:", estado_oculto_estimado)
