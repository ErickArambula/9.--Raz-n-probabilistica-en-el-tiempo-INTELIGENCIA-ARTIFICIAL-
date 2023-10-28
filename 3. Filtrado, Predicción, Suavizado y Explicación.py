import numpy as np
import matplotlib.pyplot as plt

# Definir el modelo de Kalman para predicción en una serie temporal
# En este ejemplo, asumiremos una serie temporal simple con una tendencia lineal
# y ruido gaussiano.

# Parámetros del modelo
T = 100  # Número de pasos de tiempo
estado_inicial = np.array([0, 0])  # [valor, velocidad]
ruido_proceso = 0.1  # Ruido del proceso
ruido_medida = 0.5  # Ruido de la medida

# Matrices de transición y observación
A = np.array([[1, 1],
              [0, 1]])  # Matriz de transición (modelo de movimiento)
H = np.array([[1, 0]])  # Matriz de observación (medida)

# Inicialización del filtro de Kalman
estado_predicho = estado_inicial
covarianza_predicha = np.eye(2)  # Matriz de covarianza inicial
historial_predicciones = []

# Simulación de la serie temporal
serie_temporal = []

for t in range(T):
    # Generar una medida sintética
    medida = np.dot(H, estado_inicial) + np.random.normal(0, ruido_medida)
    serie_temporal.append(medida)

    # Predicción con el filtro de Kalman
    estado_predicho = np.dot(A, estado_predicho)
    covarianza_predicha = np.dot(np.dot(A, covarianza_predicha), A.T) + np.eye(2) * ruido_proceso

    historial_predicciones.append(estado_predicho[0])

# Visualización de la serie temporal y predicciones
plt.figure(figsize=(10, 4))
plt.plot(serie_temporal, label='Medidas')
plt.plot(historial_predicciones, label='Predicciones')
plt.title("Filtro de Kalman para Predicción en Serie Temporal")
plt.xlabel("Tiempo")
plt.legend()
plt.grid(True)
plt.show()
