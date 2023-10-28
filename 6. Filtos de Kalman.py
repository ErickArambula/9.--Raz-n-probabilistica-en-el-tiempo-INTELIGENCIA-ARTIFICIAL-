import numpy as np
import matplotlib.pyplot as plt

# Modelo dinámico del sistema
A = np.array([[1, 1],
              [0, 1]])  # Matriz de transición de estado

# Matriz de covarianza del proceso
Q = np.array([[0.01, 0],
              [0, 0.01]])

# Modelo de observación
H = np.array([[1, 0]])  # Matriz de observación

# Matriz de covarianza de la observación
R = np.array([[1]])  # Ruido en la observación

# Estado inicial
estado_inicial = np.array([0, 0])  # [posición, velocidad]
covarianza_inicial = np.array([[0.1, 0],
                               [0, 0.1]])

# Tiempo y pasos de simulación
T = 50
pasos = np.arange(0, T)

# Simulación del proceso
estado_verdadero = np.zeros((T, 2))
observaciones = np.zeros((T, 1))

for t in pasos:
    # Generar estado verdadero
    if t > 0:
        estado_verdadero[t] = np.dot(A, estado_verdadero[t - 1]) + np.random.multivariate_normal([0, 0], Q)
    observaciones[t] = np.dot(H, estado_verdadero[t]) + np.random.multivariate_normal([0], R)

# Filtro de Kalman
estado_filtrado = np.zeros((T, 2))
covarianza_filtrada = np.zeros((T, 2, 2))

for t in pasos:
    if t == 0:
        estado_estimado = estado_inicial
        covarianza_estimada = covarianza_inicial
    else:
        # Predicción
        estado_predicho = np.dot(A, estado_estimado)
        covarianza_predicha = np.dot(np.dot(A, covarianza_estimada), A.T) + Q

        # Actualización (corrección)
        innovacion = observaciones[t] - np.dot(H, estado_predicho)
        innovacion_cov = R + np.dot(np.dot(H, covarianza_predicha), H.T)
        ganancia_kalman = np.dot(np.dot(covarianza_predicha, H.T), np.linalg.inv(innovacion_cov))

        estado_estimado = estado_predicho + np.dot(ganancia_kalman, innovacion)
        covarianza_estimada = covarianza_predicha - np.dot(np.dot(ganancia_kalman, H), covarianza_predicha)

    estado_filtrado[t] = estado_estimado
    covarianza_filtrada[t] = covarianza_estimada

# Visualización de resultados
plt.figure(figsize=(12, 6))
plt.plot(pasos, estado_verdadero[:, 0], label='Posición Verdadera')
plt.plot(pasos, observaciones[:, 0], 'ro', label='Observaciones')
plt.plot(pasos, estado_filtrado[:, 0], label='Posición Filtrada (Kalman)')
plt.fill_between(pasos, estado_filtrado[:, 0] - np.sqrt(covarianza_filtrada[:, 0, 0]), estado_filtrado[:, 0] + np.sqrt(covarianza_filtrada[:, 0, 0]), alpha=0.2)
plt.title("Filtro de Kalman para Estimación de Posición")
plt.xlabel("Tiempo")
plt.legend()
plt.grid(True)
plt.show()
