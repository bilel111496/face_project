import streamlit as st
import face_recognition
import os
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Face Recognition App", layout="centered")

st.title("🔍 تطبيق التعرف على الوجوه")
st.markdown("قم بتحميل صورة، وسنقوم بمحاولة التعرف على الأشخاص الموجودين فيها.")

# تحميل الوجوه المعروفة
known_faces_dir = "known_faces"
known_encodings = []
known_names = []

for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0]
            known_names.append(name)

# رفع صورة من المستخدم
uploaded_file = st.file_uploader("اختر صورة (jpg أو png)", type=["jpg", "png"])

if uploaded_file is not None:
    # قراءة الصورة وتحويلها إلى مصفوفة NumPy
    img = Image.open(uploaded_file)
    img_np = np.array(img)

    st.image(img, caption="الصورة الأصلية", use_column_width=True)

    # تحويل إلى RGB إذا كانت BGR
    if img_np.shape[2] == 3:
        rgb_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    else:
        rgb_img = img_np

    # التعرف على الوجوه
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "غير معروف"

        if True in matches:
            index = matches.index(True)
            name = known_names[index]

        # رسم مربع واسم
        cv2.rectangle(rgb_img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(rgb_img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    st.image(rgb_img, caption="النتائج", use_column_width=True)
