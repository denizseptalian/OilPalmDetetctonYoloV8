import streamlit as st
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from collections import Counter

def load_model(model_path):
    model = YOLO(model_path)
    return model

def predict_image(model, image):
    results = model(image)
    return results

def draw_results(image, results):
    img = np.array(image)
    class_counts = Counter()
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            label = f"{result.names[class_id]}: {box.conf[0]:.2f}"
            class_counts[result.names[class_id]] += 1
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img, class_counts

st.title("Deteksi dan Klasifikasi Kematangan Buah Sawit")

# Tampilkan ilustrasi gambar
st.image("Buah-Kelapa-Sawit.jpg", use_column_width=True)

# Pilihan model
model_option = st.selectbox("Pilih model deteksi:", ["Indoor (best.pt)", "Outdoor (best1.pt)"])
model_path = "best.pt" if model_option == "Indoor (best.pt)" else "best1.pt"

uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)
    
    if st.button("Predict"):  # Tombol untuk memulai prediksi
        model = load_model(model_path)  # Load model saat tombol ditekan
        results = predict_image(model, image)
        processed_image, class_counts = draw_results(image, results)
        
        st.image(processed_image, use_column_width=True)
        
        st.subheader("Perhitungan Kelas Terdeteksi")
        for class_name, count in class_counts.items():
            st.write(f"{class_name}: {count}")
