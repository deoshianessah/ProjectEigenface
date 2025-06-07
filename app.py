import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
from utils import load_images_from_folder  # pastikan versi yang diperbarui
from eigenface import EigenFaceRecognizer

st.set_page_config(layout="wide")
st.title("Pengenalan Wajah dengan PCA (EigenFace)")

# Sidebar: Input dataset
st.sidebar.header("Dataset")
dataset_path = st.sidebar.text_input("Masukkan path folder dataset", "./dataset")

if dataset_path:
    with st.spinner("Memuat dataset..."):
        images, labels, label_names, color_images = load_images_from_folder(dataset_path)

    if len(labels) == 0:
        st.error("Gagal memuat dataset. Tidak ada gambar ditemukan.")
    else:
        st.success(f"Dataset berhasil dimuat: {len(labels)} gambar")
        recognizer = EigenFaceRecognizer(images, num_components=10)

        # Upload gambar uji
        st.sidebar.header("Gambar Uji")
        uploaded_file = st.sidebar.file_uploader("Unggah gambar wajah uji", type=["jpg", "png", "jpeg"])

        if uploaded_file:
            # Baca gambar asli BERWARNA
            image_color = Image.open(uploaded_file)

            # Konversi ke grayscale untuk proses eigenface
            image_gray = image_color.convert("L")
            img_resized = np.array(image_gray.resize((100, 100))).flatten()

            # Proses pengenalan
            index, distance = recognizer.recognize(img_resized)

            st.subheader("Hasil Pengenalan:")
            if distance < 1000:
                name = label_names[labels[index]]
                st.success(f"Wajah dikenali sebagai: {name} (jarak: {distance:.2f})")

                matched_color_image = color_images[index]

                col1, col2 = st.columns(2)
                with col1:
                    st.image(image_color, caption="Gambar Uji", use_container_width=True)
                with col2:
                    st.image(matched_color_image, caption=f"Gambar Dikenali: {name}", use_container_width=True)

                st.text(f"Distance: {distance:.2f}")

            else:
                st.warning("Tidak ditemukan wajah yang mirip.")

                matched_color_image = color_images[index]

                col1, col2 = st.columns(2)
                with col1:
                    st.image(image_color, caption="Gambar Uji", use_container_width=True)
                with col2:
                    st.image(matched_color_image, caption="Wajah dari dataset yang paling mirip", use_container_width=True)

                st.text(f"Distance: {distance:.2f}")

            # Visualisasi Eigenfaces
            st.subheader("Visualisasi Komponen PCA (Eigenfaces)")
            eigenfaces = recognizer.get_eigenfaces(num_faces=5)
            cols = st.columns(len(eigenfaces))
            for i, face in enumerate(eigenfaces):
                with cols[i]:
                    st.image(face, caption=f"Eigenface {i+1}", use_container_width=True)