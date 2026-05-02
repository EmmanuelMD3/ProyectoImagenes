from pathlib import Path
import csv

CARPETA_IMAGENES = Path("dataset/imagenes")
ARCHIVO_CSV = Path("dataset/etiquetas.csv")

CLASES = [
    "manoBoca",
    "manoCara",
    "movimientoPierna",
    "desvioMirada",
    "posturaNeutral"
]

extensiones = [".jpg", ".jpeg", ".png", ".webp"]

imagenes = [
    img for img in CARPETA_IMAGENES.iterdir()
    if img.suffix.lower() in extensiones
]

with open(ARCHIVO_CSV, mode="w", newline="", encoding="utf-8") as archivo:
    writer = csv.writer(archivo)

    writer.writerow(["archivo"] + CLASES)

    for img in sorted(imagenes):
        nombre = img.name
        nombre_lower = nombre.lower()

        etiquetas = []

        for clase in CLASES:
            if nombre_lower.startswith(clase.lower()):
                etiquetas.append(1)
            else:
                etiquetas.append(0)

        writer.writerow([nombre] + etiquetas)

print(f"CSV generado correctamente: {ARCHIVO_CSV}")
print(f"Total de imágenes etiquetadas: {len(imagenes)}")