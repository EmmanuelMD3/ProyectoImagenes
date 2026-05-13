import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# ==========================
# CONFIGURACIÓN
# ==========================

# Si manoCara detecta aunque la mano esté lejos, baja a 0.10
# Si no detecta cuando sí te tocas la cara, sube a 0.15
UMBRAL_MANO_CARA = 0.13

# Si desviacionMirada se activa de frente, sube a 0.22 o 0.25
# Si no detecta cuando miras a un lado, baja a 0.15
UMBRAL_MIRADA_X = 0.18

HISTORIAL_FRAMES = 10

historial_mano_cara = deque(maxlen=HISTORIAL_FRAMES)
historial_desviacion = deque(maxlen=HISTORIAL_FRAMES)

# ==========================
# MEDIAPIPE
# ==========================

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================
# FUNCIONES
# ==========================

def distancia(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def detectar_mano_cara(face_landmarks, hand_landmarks):
    """
    Detecta mano-cara si algún punto importante de la mano
    está cerca de puntos clave del rostro.
    """

    if face_landmarks is None or hand_landmarks is None:
        return False, 1.0

    rostro = face_landmarks.landmark

    puntos_rostro = [
        rostro[1],    # nariz
        rostro[13],   # labio superior
        rostro[14],   # labio inferior
        rostro[152],  # mentón
        rostro[234],  # mejilla izquierda
        rostro[454],  # mejilla derecha
    ]

    mano = hand_landmarks.landmark

    puntos_mano = [
        mano[0],   # muñeca
        mano[4],   # pulgar
        mano[8],   # índice
        mano[12],  # medio
        mano[16],  # anular
        mano[20],  # meñique
    ]

    distancia_minima = 1.0

    for punto_mano in puntos_mano:
        for punto_rostro in puntos_rostro:
            d = distancia(punto_mano, punto_rostro)
            distancia_minima = min(distancia_minima, d)

    mano_cara = distancia_minima <= UMBRAL_MANO_CARA

    return mano_cara, distancia_minima


def detectar_desviacion_mirada(face_landmarks):
    """
    Detecta desviación de mirada usando la posición horizontal del iris.
    Si el iris se aleja del centro del ojo, se marca desviación.
    """

    if face_landmarks is None:
        return False, 0.0

    lm = face_landmarks.landmark

    try:
        # Ojo izquierdo: extremos 33 y 133, iris 468
        ojo_izq_1 = lm[33]
        ojo_izq_2 = lm[133]
        iris_izq = lm[468]

        # Ojo derecho: extremos 362 y 263, iris 473
        ojo_der_1 = lm[362]
        ojo_der_2 = lm[263]
        iris_der = lm[473]

    except IndexError:
        return False, 0.0

    def posicion_iris(extremo1, extremo2, iris):
        x_min = min(extremo1.x, extremo2.x)
        x_max = max(extremo1.x, extremo2.x)

        if x_max - x_min == 0:
            return 0.5

        return (iris.x - x_min) / (x_max - x_min)

    pos_izq = posicion_iris(ojo_izq_1, ojo_izq_2, iris_izq)
    pos_der = posicion_iris(ojo_der_1, ojo_der_2, iris_der)

    promedio = (pos_izq + pos_der) / 2

    # 0.5 representa el centro aproximado del ojo
    score = abs(promedio - 0.5)

    desviacion = score >= UMBRAL_MIRADA_X

    return desviacion, score


def dibujar_texto(frame, texto, y, detectado):
    if detectado:
        color = (0, 255, 0)
        texto = texto + " DETECTADO"
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
        print("No se pudo leer frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    resultado_face = face_mesh.process(frame_rgb)
    resultado_hands = hands.process(frame_rgb)

    hay_rostro = resultado_face.multi_face_landmarks is not None

    if not hay_rostro:
        historial_mano_cara.clear()
        historial_desviacion.clear()

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
            "No se analizan conductas",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    else:
        face_landmarks = resultado_face.multi_face_landmarks[0]

        # ==========================
        # DESVIACIÓN DE MIRADA
        # ==========================

        desviacion, score_mirada = detectar_desviacion_mirada(face_landmarks)

        # ==========================
        # MANO A CARA
        # ==========================

        mano_cara_detectada = False
        distancia_mano = 1.0

        if resultado_hands.multi_hand_landmarks:
            for hand_landmarks in resultado_hands.multi_hand_landmarks:
                mano_cara, d = detectar_mano_cara(face_landmarks, hand_landmarks)

                if mano_cara:
                    mano_cara_detectada = True

                distancia_mano = min(distancia_mano, d)

        # ==========================
        # SUAVIZADO TEMPORAL
        # ==========================

        historial_mano_cara.append(1 if mano_cara_detectada else 0)
        historial_desviacion.append(1 if desviacion else 0)

        mano_prom = np.mean(historial_mano_cara)
        mirada_prom = np.mean(historial_desviacion)

        mano_final = mano_prom >= 0.6
        desviacion_final = mirada_prom >= 0.6

        # Postura neutral calculada:
        # si hay rostro y no hay conductas activas
        postura_neutral = not mano_final and not desviacion_final

        # ==========================
        # MOSTRAR TEXTO
        # ==========================

        y = 35

        dibujar_texto(
            frame,
            f"manoCara: dist={distancia_mano:.2f}",
            y,
            mano_final
        )
        y += 35

        dibujar_texto(
            frame,
            f"desviacionMirada: score={score_mirada:.2f}",
            y,
            desviacion_final
        )
        y += 35

        dibujar_texto(
            frame,
            "posturaNeutral",
            y,
            postura_neutral
        )

    cv2.putText(
        frame,
        "Presiona 'q' para salir",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow("Sistema MediaPipe - Inquietud Observable", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()