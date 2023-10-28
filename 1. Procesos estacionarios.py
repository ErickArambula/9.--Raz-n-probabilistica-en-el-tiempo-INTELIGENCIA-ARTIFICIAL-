import numpy as np
import matplotlib.pyplot as plt

# Generar un proceso de ruido blanco
np.random.seed(0)  # Fijar la semilla para reproducibilidad
muestras = 1000
ruido_blanco = np.random.normal(0, 1, muestras)

# Visualizar el proceso de ruido blanco
plt.figure(figsize=(10, 4))
plt.plot(ruido_blanco)
plt.title("Proceso de Ruido Blanco")
plt.xlabel("Tiempo")
plt.ylabel("Valor")
plt.grid(True)
plt.show()
