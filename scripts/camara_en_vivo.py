import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import deque

# ==========================
# CONFIGURACIÓN
# ==========================

RUTA_MODELO = Path("modelos/mejor_modelo_multilabel.keras")

CLASES = [
    "manoCara",
    "desviacionMirada",
    "posturaNeutral"
]

IMG_SIZE = (224, 224)

UMBRALES = {
    "manoCara": 0.75,
    "desviacionMirada": 0.80,
    "posturaNeutral": 0.80
}

historial = {
    clase: deque(maxlen=20) for clase in CLASES
}

# ==========================
# CARGAR MODELO
# ==========================

modelo = tf.keras.models.load_model(RUTA_MODELO)

print("Modelo cargado desde:", RUTA_MODELO)
print("Forma de salida del modelo:", modelo.output_shape)
print("Clases en cámara:", CLASES)

if modelo.output_shape[-1] != len(CLASES):
    raise ValueError(
        f"El modelo tiene {modelo.output_shape[-1]} salidas, "
        f"pero en CLASES tienes {len(CLASES)} clases."
    )

# Historial para suavizar predicciones
# Guarda las últimas 10 predicciones de cada clase
historial = {
    clase: deque(maxlen=20) for clase in CLASES
}

# ==========================
# CARGAR MODELO
# ==========================

modelo = tf.keras.models.load_model(RUTA_MODELO)

# ==========================
# CÁMARA
# ==========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

print("Cámara iniciada. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("No se pudo leer frame de la cámara.")
        break

    # Convertir BGR a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Redimensionar
    img = cv2.resize(frame_rgb, IMG_SIZE)

    # Normalizar 0-1
    img = img.astype("float32") / 255.0

    # Agregar dimensión batch
    img = np.expand_dims(img, axis=0)

    # Predicción
    predicciones = modelo.predict(img, verbose=0)[0]

    # Guardar predicciones en historial
    for clase, prob in zip(CLASES, predicciones):
        historial[clase].append(prob)

    y = 30

    for clase in CLASES:
        # Promedio de las últimas predicciones
        prob_promedio = np.mean(historial[clase])

        texto = f"{clase}: {prob_promedio:.2f}"

        if prob_promedio >= UMBRALES[clase]:
            color = (0, 255, 0)
            estado = " DETECTADO"
        else:
            color = (0, 0, 255)
            estado = ""

        cv2.putText(
            frame,
            texto + estado,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        y += 35

    # Mostrar información de ayuda
    cv2.putText(
        frame,
        "Presiona 'q' para salir",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow("Deteccion en vivo - CNN Multilabel", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()