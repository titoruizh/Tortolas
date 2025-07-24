# Databricks notebook source
from ultralytics import YOLO
import os

def entrenar_yolo():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    print("🚀 Iniciando entrenamiento...")

    model = YOLO('yolov8m-seg.pt')  # o usa best.pt si quieres reentrenar

    model.train(
        data="data.yaml",  # asegúrate de que esté en la misma carpeta
        epochs=50,
        imgsz=1024,
        batch=1,
        name="modelo_tortolas_v1_seg_desde_web",
        project="RUN",  # ✅ ruta relativa
        device="cpu",   # o "cuda" si tienes GPU
        workers=0,
        patience=20,
        cache=False
    )

    return "✅ Entrenamiento completado con éxito."