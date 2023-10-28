from hmmlearn import hmm
import numpy as np

# Definir el modelo HMM
modelo_hmm = hmm.MultinomialHMM(n_components=2)

# Definir la matriz de transición de estado
modelo_hmm.transmat_ = np.array([[0.7, 0.3],
                                 [0.4, 0.6]])

# Definir las probabilidades de emisión
modelo_hmm.emissionprob_ = np.array([[0.8, 0.2],
                                     [0.3, 0.7]])

# Generar una secuencia de observaciones sintéticas
secuencia_observaciones, secuencia_estados = modelo_hmm.sample(100)

# Estimar los estados ocultos a partir de las observaciones
estados_estimados = modelo_hmm.predict(secuencia_observaciones)

print("Secuencia de Observaciones:")
print(secuencia_observaciones)
print("Secuencia de Estados Ocultos Estimados:")
print(estados_estimados)
