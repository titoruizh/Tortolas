from ultralytics import YOLO
import os
from PIL import Image
import numpy as np
import yaml

def cargar_nombres_desde_yaml(yaml_path="data.yaml"):
    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return data.get("names", [])
    except Exception as e:
        print(f"❌ Error al cargar nombres desde {yaml_path}: {e}")
        return []


def detectar_objetos(imagen, modelo_path="yolov8_model/yolov8m-seg.pt", yaml_path="data.yaml"):
    model = YOLO(modelo_path)
    results = model.predict(imagen)

    nombres_personalizados = cargar_nombres_desde_yaml(yaml_path)

    for r in results:
        if nombres_personalizados and len(nombres_personalizados) == model.model.nc:
            r.names = {i: name for i, name in enumerate(nombres_personalizados)}
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        return im

def entrenar_modelo_yolo(yaml_path="data.yaml", epochs=50, project_path="RUN", nombre_modelo="modelo_entrenado"):
    try:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        print("🚀 Iniciando entrenamiento...")

        model = YOLO("yolov8m-seg.pt")
        model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=1024,
            batch=1,
            name=nombre_modelo,
            project=project_path,
            device="cpu",
            workers=0,
            patience=20,
            cache=False
        )

        return "✅ Entrenamiento completado con éxito."

    except Exception as e:
        return f"❌ Error durante el entrenamiento: {str(e)}"

def get_last_epoch(project_folder):
    try:
        folders = [f for f in os.listdir(project_folder) if os.path.isdir(os.path.join(project_folder, f))]
        folders.sort(reverse=True)
        last = folders[0] if folders else None
        return last
    except Exception:
        return None