import os
import streamlit as st

import cv2
import supervision as sv

from tqdm import tqdm
from inference.models.yolo_world.yolo_world import YOLOWorld

import requests
import numpy as np


def save_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        with open("uploads/image.jpeg", "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        predict()
        
def predict():
    
    # Set image path
    SOURCE_IMAGE_PATH = f"uploads/image.jpeg"
    
    # Retrieve model model
    model = YOLOWorld(model_id="yolo_world/l")
    
    # Set prediction classes
    classes = ["person", "backpack", "dog", "eye", "nose", "ear", "tongue"]
    model.set_classes(classes)

    
    image = cv2.imread(SOURCE_IMAGE_PATH)
    results = model.infer(image, confidence=0.003)
    detections = sv.Detections.from_inference(results).with_nms(threshold=0.1)
    
    
    BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
    LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)

    labels = [
        f"{classes[class_id]} {confidence:0.3f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    annotated_image = image.copy()
    annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections)
    annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections, labels=labels)
    
    
    
    
    # Show title
    st.title("Object Detection Results")
    
    # Show image with annotation of predicted objects
    annotated_image = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)
    st.image(annotated_image, use_column_width=True)

def main():
    
    # sidebar title
    st.sidebar.title('Object Detection using YOLO World')

    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if st.sidebar.button("Predict Objects") and uploaded_file is not None:
        save_uploaded_file(uploaded_file)

if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    main()
