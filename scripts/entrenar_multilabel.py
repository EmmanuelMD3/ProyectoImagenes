import os
from pathlib import Path

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


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
EPOCHS = 30
SEED = 123


# ==========================
# CARGAR CSV
# ==========================

df = pd.read_csv(RUTA_CSV)

if "archivo" not in df.columns:
    raise ValueError("El archivo etiquetas.csv debe tener una columna llamada 'archivo'.")

CLASES = [col for col in df.columns if col != "archivo"]

print("Clases detectadas:", CLASES)
print("Total de registros en CSV:", len(df))

df["ruta"] = df["archivo"].apply(lambda x: str(RUTA_IMAGENES / x))

df = df[df["ruta"].apply(os.path.exists)].reset_index(drop=True)

print("Imágenes encontradas:", len(df))

if len(df) == 0:
    raise ValueError("No se encontraron imágenes. Revisa dataset/imagenes y etiquetas.csv.")

print("\nDistribución de etiquetas:")
print(df[CLASES].sum())


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

print("\nDivisión del dataset:")
print("Entrenamiento:", len(train_df))
print("Validación:", len(val_df))
print("Prueba:", len(test_df))


# ==========================
# CARGAR IMÁGENES
# ==========================

def cargar_imagen(ruta, etiquetas):
    imagen = tf.io.read_file(ruta)
    imagen = tf.image.decode_image(
        imagen,
        channels=3,
        expand_animations=False
    )
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
# MODELO MULTILABEL CON MOBILENETV2
# ==========================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10),
], name="data_augmentation")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

# Las imágenes vienen en 0-1, MobileNetV2 espera el preprocesamiento sobre 0-255
x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)

# MULTILABEL: una salida independiente por conducta
outputs = layers.Dense(len(CLASES), activation="sigmoid")(x)

modelo = tf.keras.Model(inputs, outputs)

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

modelo.summary()


# ==========================
# CALLBACKS
# ==========================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    filepath=RUTA_MODELOS / "mejor_modelo_multilabel.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)


# ==========================
# ENTRENAMIENTO
# ==========================

historial = modelo.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)


# ==========================
# EVALUACIÓN
# ==========================

resultados = modelo.evaluate(test_ds)

print("\nResultados en prueba:")
nombres_metricas = ["loss", "binary_accuracy", "precision", "recall"]

for nombre, valor in zip(nombres_metricas, resultados):
    print(f"{nombre}: {valor:.4f}")


# ==========================
# GUARDAR MODELO FINAL
# ==========================

modelo.save(RUTA_MODELOS / "modelo_multilabel.keras")

print("\nModelo final guardado en:")
print(RUTA_MODELOS / "modelo_multilabel.keras")

print("\nMejor modelo guardado en:")
print(RUTA_MODELOS / "mejor_modelo_multilabel.keras")


# ==========================
# GUARDAR HISTORIAL
# ==========================

historial_df = pd.DataFrame(historial.history)
historial_df.to_csv(
    RUTA_RESULTADOS / "historial_entrenamiento.csv",
    index=False
)


# ==========================
# GUARDAR RESULTADOS
# ==========================

resultados_df = pd.DataFrame({
    "metrica": nombres_metricas[:len(resultados)],
    "valor": resultados
})

resultados_df.to_csv(
    RUTA_RESULTADOS / "matriz_resultados.csv",
    index=False
)


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


print("\nGráficas guardadas en:")
print(RUTA_GRAFICAS)

print("\nDistribución de etiquetas:")
print(df[CLASES].sum())