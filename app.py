import os
import base64
import streamlit as st
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO
from utils import detectar_objetos, entrenar_modelo_yolo, get_last_epoch

st.set_page_config(
    page_title="Análisis de Ortomosaicos - Minera Tórtolas",
    page_icon="🛰️",
    layout="centered",
)

def mostrar_logo_fondo(logo_path="logo_mina.png", ancho_px=300, opacidad=0.06):
    if not os.path.exists(logo_path):
        st.warning(f"⚠️ No se encontró el archivo '{logo_path}'.")
        return
    with open(logo_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <div style="
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            opacity: {opacidad};
            z-index: 0;
            pointer-events: none;
        ">
            <img src="data:image/png;base64,{encoded}" width="{ancho_px}">
        </div>
        """,
        unsafe_allow_html=True,
    )

mostrar_logo_fondo("logo_mina.png", ancho_px=1000, opacidad=0.07)

st.title("🔍 Análisis automático de ortomosaicos")
st.markdown("""
Este sistema analiza ortomosaicos capturados por drones en faenas mineras.
Detecta automáticamente objetos como tuberías, vehículos o caminos mediante inteligencia artificial.
""")

modelo_path = "yolov8_model/best.pt"
yaml_path = "data.yaml"

if not os.path.exists(modelo_path):
    st.error("❌ No se encontró el modelo entrenado (`best.pt`). Verifica la ruta.")
    st.stop()

modelo = YOLO(modelo_path)
os.makedirs("test_images", exist_ok=True)

tab1, tab2, tab3 = st.tabs(["📤 Subir imagen", "📈 Información del modelo", "🛠️ Entrenar modelo"])


with tab1:
    archivo = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png", "tif"])
    if archivo:
        try:
            imagen = Image.open(archivo)
            st.image(imagen, caption="🖼️ Imagen original", use_container_width=True)
            ruta_imagen = os.path.join("test_images", archivo.name)
            imagen.save(ruta_imagen)

            if st.button("🚀 Analizar imagen"):
                with st.spinner("Detectando objetos..."):
                    resultado = detectar_objetos(ruta_imagen, modelo_path=modelo_path, yaml_path=yaml_path)
                    resultado.save("resultado.jpg")
                    st.image("resultado.jpg", caption="📌 Resultado segmentado", use_container_width=True)

        except UnidentifiedImageError:
            st.error("❌ El archivo seleccionado no es una imagen válida.")

with tab2:
    st.markdown("""
    ### Modelo YOLOv8 personalizado
    - 🔧 Base: `yolov8m-seg`
    - 🛰️ Dataset: Ortomosaicos del tranque *Las Tórtolas*
    - 🎯 Clases: Automáticamente cargadas desde `data.yaml`
    - 📊 Precisión estimada: mAP > 87%
    """)

with tab3:
    st.markdown("### 🏗️ Entrenar modelo desde cero")
    ruta_yaml = st.text_input("📄 Ruta del archivo .yaml del dataset", "data.yaml")
    epochs = st.number_input("🔁 Número de epochs", min_value=1, max_value=500, value=50)

    if st.button("🚀 Iniciar entrenamiento"):
        try:
            with st.spinner("Entrenando modelo... esto puede tardar varios minutos"):
                entrenar_modelo_yolo(ruta_yaml, epochs=epochs)
            st.success("✅ Entrenamiento finalizado correctamente.")
        except Exception as e:
            st.error(f"❌ Ocurrió un error durante el entrenamiento:\n\n`{e}`")

st.markdown("---")
st.caption("🛠️ Proyecto desarrollado por Florentino Vargas / Tito Ruiz - Magíster en Ingeniería Informática")
