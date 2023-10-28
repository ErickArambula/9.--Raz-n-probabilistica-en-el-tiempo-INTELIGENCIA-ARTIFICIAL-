import speech_recognition as sr

# Crear un reconocedor de voz
recognizer = sr.Recognizer()

# Configurar el motor de reconocimiento de voz
recognizer.energy_threshold = 4000  # Umbral de energía para activar el reconocimiento

# Grabar audio desde un archivo o dispositivo de audio
with sr.AudioFile("ejemplo.wav") as fuente:
    audio = recognizer.listen(fuente)

try:
    # Reconocer el habla utilizando el motor CMU Sphinx
    texto_reconocido = recognizer.recognize_sphinx(audio)
    print("Texto reconocido:", texto_reconocido)

except sr.UnknownValueError:
    print("No se pudo entender el habla")

except sr.RequestError as e:
    print("Error en la solicitud: {0}".format(e))
