import numpy as np
from filterpy.monte_carlo import multinomial_resample
from filterpy.kalman import KalmanFilter
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

# Número de partículas
num_particulas = 100

# Generar partículas iniciales
particulas = np.random.multivariate_normal(estado_inicial, covarianza_inicial, num_particulas)

# Simulación del proceso
T = 50
posicion_verdadera = np.zeros(T)
observaciones = np.zeros(T)

for t in range(T):
    # Generar estado verdadero
    if t > 0:
        estado_verdadero = np.dot(A, estado_verdadero) + np.random.multivariate_normal([0, 0], Q)
    posicion_verdadera[t] = estado_verdadero[0]
    observaciones[t] = np.dot(H, estado_verdadero) + np.random.multivariate_normal([0], R)

# Filtrado de partículas
predicciones = np.zeros((T, num_particulas))
pesos = np.ones(num_particulas) / num_particulas

for t in range(T):
    # Predicción de las partículas
    for i in range(num_particulas):
        particula = particulas[i]
        particula = np.dot(A, particula) + np.random.multivariate_normal([0, 0], Q)
        predicciones[t, i] = particula[0]
        particulas[i] = particula

    # Actualización de los pesos de las partículas
    innovacion = observaciones[t] - np.dot(H, particulas.T)
    likelihood = np.exp(-0.5 * (innovacion ** 2 / R))
    pesos *= likelihood
    pesos += 1.e-300  # Evitar división por cero
    pesos /= sum(pesos)

    # Re-muestreo de partículas
    indices = multinomial_resample(pesos)
    particulas = particulas[indices]

# Visualización de resultados
plt.figure(figsize=(12, 6))
plt.plot(posicion_verdadera, label='Posición Verdadera')
plt.plot(observaciones, 'ro', label='Observaciones')
plt.plot(predicciones, 'g', alpha=0.5, label='Predicciones (Partículas)')
plt.title("Filtrado de Partículas para Estimación de Posición")
plt.xlabel("Tiempo")
plt.legend()
plt.grid(True)
plt.show()
