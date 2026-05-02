import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# ==========================
# CONFIGURACIÓN
# ==========================

RUTA_DATASET = Path("dataset")
RUTA_IMAGENES = RUTA_DATASET / "imagenes"
RUTA_CSV = RUTA_DATASET / "etiquetas.csv"

RUTA_MODELOS = Path("modelos")
RUTA_RESULTADOS = Path("resultados")
RUTA_GRAFICAS = RUTA_RESULTADOS / "graficas"

RUTA_MODELOS.mkdir(exist_ok=True)
RUTA_RESULTADOS.mkdir(exist_ok=True)
RUTA_GRAFICAS.mkdir(exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 20
SEED = 123

# ==========================
# CARGAR CSV
# ==========================

df = pd.read_csv(RUTA_CSV)

# Todas las columnas excepto "archivo" son etiquetas
CLASES = [col for col in df.columns if col != "archivo"]

print("Clases detectadas:", CLASES)
print("Total de imágenes:", len(df))

# Crear ruta completa de cada imagen
df["ruta"] = df["archivo"].apply(lambda x: str(RUTA_IMAGENES / x))

# Verificar que existan las imágenes
df = df[df["ruta"].apply(os.path.exists)].reset_index(drop=True)

print("Imágenes encontradas:", len(df))

# ==========================
# DIVISIÓN TRAIN / VAL / TEST
# ==========================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=SEED,
    shuffle=True
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    shuffle=True
)

print("Entrenamiento:", len(train_df))
print("Validación:", len(val_df))
print("Prueba:", len(test_df))

# ==========================
# FUNCIÓN PARA CARGAR IMAGEN
# ==========================

def cargar_imagen(ruta, etiquetas):
    imagen = tf.io.read_file(ruta)
    imagen = tf.image.decode_image(imagen, channels=3, expand_animations=False)
    imagen = tf.image.resize(imagen, IMG_SIZE)
    imagen = tf.cast(imagen, tf.float32) / 255.0
    return imagen, etiquetas

def crear_dataset(dataframe, shuffle=True):
    rutas = dataframe["ruta"].values
    etiquetas = dataframe[CLASES].values.astype("float32")

    ds = tf.data.Dataset.from_tensor_slices((rutas, etiquetas))
    ds = ds.map(cargar_imagen, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe), seed=SEED)

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

train_ds = crear_dataset(train_df, shuffle=True)
val_ds = crear_dataset(val_df, shuffle=False)
test_ds = crear_dataset(test_df, shuffle=False)

# ==========================
# MODELO CNN MULTILABEL
# ==========================

modelo = models.Sequential([
    layers.Input(shape=(224, 224, 3)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    # MULTILABEL:
    # Una neurona por conducta, con sigmoid independiente
    layers.Dense(len(CLASES), activation="sigmoid")
])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        "binary_accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

modelo.summary()

# ==========================
# ENTRENAMIENTO
# ==========================

historial = modelo.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ==========================
# EVALUACIÓN
# ==========================

resultados = modelo.evaluate(test_ds)

print("\nResultados en prueba:")
for nombre, valor in zip(modelo.metrics_names, resultados):
    print(f"{nombre}: {valor:.4f}")

# ==========================
# GUARDAR MODELO
# ==========================

modelo.save(RUTA_MODELOS / "modelo_multilabel.keras")
print("\nModelo guardado en modelos/modelo_multilabel.keras")

# ==========================
# GUARDAR HISTORIAL
# ==========================

historial_df = pd.DataFrame(historial.history)
historial_df.to_csv(RUTA_RESULTADOS / "historial_entrenamiento.csv", index=False)

# ==========================
# GUARDAR RESULTADOS
# ==========================

nombres_metricas = ["loss", "binary_accuracy", "precision", "recall"]

resultados_df = pd.DataFrame({
    "metrica": nombres_metricas[:len(resultados)],
    "valor": resultados
})

resultados_df.to_csv(RUTA_RESULTADOS / "matriz_resultados.csv", index=False)

# ==========================
# GRÁFICA ACCURACY
# ==========================

plt.figure()
plt.plot(historial.history["binary_accuracy"], label="Entrenamiento")
plt.plot(historial.history["val_binary_accuracy"], label="Validación")
plt.title("Precisión del modelo multilabel")
plt.xlabel("Épocas")
plt.ylabel("Binary Accuracy")
plt.legend()
plt.savefig(RUTA_GRAFICAS / "accuracy.png")
plt.close()

# ==========================
# GRÁFICA LOSS
# ==========================

plt.figure()
plt.plot(historial.history["loss"], label="Entrenamiento")
plt.plot(historial.history["val_loss"], label="Validación")
plt.title("Pérdida del modelo multilabel")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.legend()
plt.savefig(RUTA_GRAFICAS / "loss.png")
plt.close()

print("\nGráficas guardadas en resultados/graficas/")