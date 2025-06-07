import os
import cv2
import numpy as np

def load_images_from_folder(folder):
    images = []
    labels = []
    label_names = []
    color_images = []
    label_map = {}

    label_index = 0
    for person_name in os.listdir(folder):
        person_path = os.path.join(folder, person_name)
        if not os.path.isdir(person_path):
            continue

        if person_name not in label_map:
            label_map[person_name] = label_index
            label_names.append(person_name)
            label_index += 1

        for filename in os.listdir(person_path):
            file_path = os.path.join(person_path, filename)

            # Baca gambar grayscale dan RGB
            img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img_color = cv2.imread(file_path, cv2.IMREAD_COLOR)

            if img_gray is not None and img_color is not None:
                img_gray_resized = cv2.resize(img_gray, (100, 100)).flatten()
                img_color_resized = cv2.resize(img_color, (100, 100))

                images.append(img_gray_resized)
                color_images.append(cv2.cvtColor(img_color_resized, cv2.COLOR_BGR2RGB))  # convert ke RGB utk Streamlit
                labels.append(label_map[person_name])

    return np.array(images).T, labels, label_names, color_images