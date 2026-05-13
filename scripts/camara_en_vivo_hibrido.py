import cv2
import numpy as np
import tensorflow as tf
import mediapipe.python.solutions.face_mesh as mp_face_mesh
from pathlib import Path
from collections import deque

# ==========================
# CONFIGURACIÓN
# ==========================

RUTA_MODELO = Path("modelos/mejor_modelo_multilabel.keras")

CLASES_CNN = [
    "manoCara",
    "desviacionMirada",
    "posturaNeutral"
]

IMG_SIZE = (224, 224)

# Umbral para CNN
UMBRAL_MANO_CARA = 0.75

# Umbral para mirada usando FaceMesh
# Si está muy sensible, sube estos valores.
UMBRAL_MIRADA_X = 0.18
UMBRAL_MIRADA_Y = 0.22

# Suavizado temporal
historial_mano = deque(maxlen=15)
historial_mirada = deque(maxlen=15)

# ==========================
# CARGAR MODELO CNN
# ==========================

modelo = tf.keras.models.load_model(RUTA_MODELO)

print("Modelo cargado desde:", RUTA_MODELO)
print("Forma de salida del modelo:", modelo.output_shape)
print("Clases CNN:", CLASES_CNN)

if modelo.output_shape[-1] != len(CLASES_CNN):
    raise ValueError(
        f"El modelo tiene {modelo.output_shape[-1]} salidas, "
        f"pero configuraste {len(CLASES_CNN)} clases."
    )

# ==========================
# MEDIAPIPE FACE MESH
# ==========================

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================
# FUNCIONES
# ==========================

def predecir_mano_cara(frame_bgr):
    """
    Usa la CNN para obtener la probabilidad de manoCara.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(frame_rgb, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    predicciones = modelo.predict(img, verbose=0)[0]

    resultados = {
        clase: float(prob)
        for clase, prob in zip(CLASES_CNN, predicciones)
    }

    return resultados["manoCara"]


def detectar_desviacion_mirada(frame_bgr):
    """
    Detecta desviación de mirada usando FaceMesh.
    Devuelve:
    - hay_rostro: True/False
    - desviacion: True/False
    - score: valor aproximado de desviación
    """

    alto, ancho, _ = frame_bgr.shape
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    resultado = face_mesh.process(frame_rgb)

    if not resultado.multi_face_landmarks:
        return False, False, 0.0

    landmarks = resultado.multi_face_landmarks[0].landmark

    # Landmarks principales con iris refinado
    # Ojo izquierdo: extremos 33 y 133, iris 468
    # Ojo derecho: extremos 362 y 263, iris 473
    try:
        ojo_izq_ext_1 = landmarks[33]
        ojo_izq_ext_2 = landmarks[133]
        iris_izq = landmarks[468]

        ojo_der_ext_1 = landmarks[362]
        ojo_der_ext_2 = landmarks[263]
        iris_der = landmarks[473]
    except IndexError:
        return True, False, 0.0

    # Posición horizontal normalizada del iris dentro de cada ojo
    def posicion_iris_x(ext1, ext2, iris):
        x_min = min(ext1.x, ext2.x)
        x_max = max(ext1.x, ext2.x)

        if x_max - x_min == 0:
            return 0.5

        return (iris.x - x_min) / (x_max - x_min)

    pos_izq = posicion_iris_x(ojo_izq_ext_1, ojo_izq_ext_2, iris_izq)
    pos_der = posicion_iris_x(ojo_der_ext_1, ojo_der_ext_2, iris_der)

    promedio_x = (pos_izq + pos_der) / 2

    # En una mirada frontal, el promedio suele estar cerca de 0.5
    desviacion_x = abs(promedio_x - 0.5)

    # También revisamos inclinación vertical aproximada usando iris respecto a nariz
    # Nariz: landmark 1
    nariz = landmarks[1]
    promedio_iris_y = (iris_izq.y + iris_der.y) / 2
    desviacion_y = abs(promedio_iris_y - nariz.y)

    # Score general
    score = max(desviacion_x, desviacion_y)

    desviacion = (
        desviacion_x >= UMBRAL_MIRADA_X or
        desviacion_y >= UMBRAL_MIRADA_Y
    )

    # Dibujar puntos principales para depurar
    puntos = [iris_izq, iris_der]
    for p in puntos:
        cx = int(p.x * ancho)
        cy = int(p.y * alto)
        cv2.circle(frame_bgr, (cx, cy), 3, (255, 255, 0), -1)

    return True, desviacion, float(score)


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

    # 1. Detectar rostro y mirada con MediaPipe
    hay_rostro, mirada_desviada, score_mirada = detectar_desviacion_mirada(frame)

    if not hay_rostro:
        # Si no hay rostro, no ejecutar CNN ni marcar conductas
        historial_mano.clear()
        historial_mirada.clear()

        cv2.putText(
            frame,
            "Sin rostro detectado",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            "No se ejecuta prediccion",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    else:
        # 2. CNN solo para manoCara
        prob_mano = predecir_mano_cara(frame)

        historial_mano.append(prob_mano)
        historial_mirada.append(1.0 if mirada_desviada else 0.0)

        mano_prom = np.mean(historial_mano)
        mirada_prom = np.mean(historial_mirada)

        mano_detectada = mano_prom >= UMBRAL_MANO_CARA
        mirada_detectada = mirada_prom >= 0.6

        # 3. Neutral calculado por lógica
        postura_neutral = not mano_detectada and not mirada_detectada

        y = 35

        # Mostrar manoCara
        texto = f"manoCara CNN: {mano_prom:.2f}"
        if mano_detectada:
            texto += " DETECTADO"
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.putText(
            frame,
            texto,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        y += 35

        # Mostrar desviacionMirada
        texto = f"desviacionMirada FaceMesh: {score_mirada:.2f}"
        if mirada_detectada:
            texto += " DETECTADO"
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.putText(
            frame,
            texto,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        y += 35

        # Mostrar postura neutral calculada
        if postura_neutral:
            texto = "posturaNeutral: CALCULADA DETECTADO"
            color = (0, 255, 0)
        else:
            texto = "posturaNeutral: CALCULADA"
            color = (0, 0, 255)

        cv2.putText(
            frame,
            texto,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    # Ayuda
    cv2.putText(
        frame,
        "Presiona 'q' para salir",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow("Sistema hibrido - CNN + FaceMesh", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()