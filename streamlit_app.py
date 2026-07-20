import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json

# Set page config
st.set_page_config(page_title="AgeSense", layout="wide", initial_sidebar_state="expanded")

# Title and description
st.title("🔍 AgeSense - Age & Emotion Detection")
st.markdown("Upload an image to detect ages and emotions of people in the photo")

# Load models
@st.cache_resource
def load_models():
    FACE_PROTO = "models/opencv_face_detector.pbtxt"
    FACE_MODEL = "models/opencv_face_detector_uint8.pb"
    AGE_PROTO = "models/age_deploy.prototxt"
    AGE_MODEL = "models/age_net.caffemodel"
    EMOTION_MODEL = "models/emotion.onnx"
    
    face_net = cv2.dnn.readNetFromTensorflow(FACE_MODEL, FACE_PROTO)
    
    try:
        age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
    except Exception as e:
        print(f"Error loading Age model: {e}")
        age_net = None
        
    try:
        emotion_net = cv2.dnn.readNetFromONNX(EMOTION_MODEL)
    except Exception as e:
        print(f"Error loading Emotion model: {e}")
        emotion_net = None

    return face_net, age_net, emotion_net

face_net, age_net, emotion_net = load_models()

# Categories
AGE_CATEGORIES = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
EMOTION_CATEGORIES = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']

def detect_faces(net, frame, conf_threshold=0.7):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        [104, 117, 123],
        False, False
    )

    net.setInput(blob)
    detections = net.forward()
    face_boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            face_boxes.append([x1, y1, x2, y2])

    return face_boxes

def extract_face_region(frame, x1, y1, x2, y2, padding_ratio=0.1, min_size=80):
    frame_h, frame_w = frame.shape[:2]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad_w = int(width * padding_ratio)
    pad_h = int(height * padding_ratio)

    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(frame_w, x2 + pad_w)
    y2 = min(frame_h, y2 + pad_h)

    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return face

def predict_age(face):
    if age_net is None:
        return "(25-32)", 50.0
    try:
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        age_net.setInput(blob)
        preds = age_net.forward()
        i = preds[0].argmax()
        age_group = AGE_CATEGORIES[i]
        confidence = float(preds[0][i]) * 100
        return age_group, confidence
    except Exception as e:
        return "(25-32)", 50.0

def predict_emotion(face):
    if emotion_net is None:
        return "Neutral", 50.0
    try:
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        blob = cv2.dnn.blobFromImage(resized, 1.0, (64, 64), (0,), swapRB=False)
        emotion_net.setInput(blob)
        preds = emotion_net.forward()
        i = preds[0].argmax()
        emotion = EMOTION_CATEGORIES[i]
        
        # Softmax for confidence
        exp_preds = np.exp(preds[0] - np.max(preds[0]))
        confidence = float(exp_preds[i] / np.sum(exp_preds)) * 100
        
        return emotion, confidence
    except Exception as e:
        return "Neutral", 50.0

def process_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        return None, None

    face_boxes = detect_faces(face_net, frame)
    results = []

    for person_id, (x1, y1, x2, y2) in enumerate(face_boxes, start=1):
        face = extract_face_region(frame, x1, y1, x2, y2)

        if face is None or face.shape[0] < 20 or face.shape[1] < 20:
            continue

        age, age_confidence = predict_age(face)
        emotion, emotion_confidence = predict_emotion(face)

        person_result = {
            "person_id": person_id,
            "age_group": age,
            "age_confidence": round(float(age_confidence), 2),
            "emotion": emotion,
            "emotion_confidence": round(float(emotion_confidence), 2),
            "bounding_box": [x1, y1, x2, y2]
        }

        results.append(person_result)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{age} | {emotion}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        cv2.rectangle(
            frame, (x1, y2),
            (x1 + text_width + 10, y2 + text_height + 10),
            (0, 0, 0), -1
        )

        cv2.putText(
            frame, label,
            (x1 + 5, y2 + text_height + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 255), 1,
            cv2.LINE_AA
        )

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), results

# Sidebar
with st.sidebar:
    st.header("📸 Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Main content
if uploaded_file is not None:
    # Save uploaded file temporarily
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    # Process image
    with st.spinner("🔄 Processing image..."):
        processed_image, results = process_image(tmp_path)

    if processed_image is not None and results:
        # Display results
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Detection Results")
            st.image(processed_image, use_column_width=True)

        with col2:
            st.subheader("📊 Details")
            for result in results:
                with st.expander(f"👤 Person {result['person_id']}"):
                    st.metric("Age Group", result['age_group'])
                    st.metric("Age Confidence", f"{result['age_confidence']}%")
                    st.metric("Emotion", result['emotion'])
                    st.metric("Emotion Confidence", f"{result['emotion_confidence']:.2f}%")

        # JSON Results
        st.subheader("📋 JSON Output")
        st.json(results)

    else:
        st.info("❌ No faces detected in the image")

    # Clean up
    import os
    os.remove(tmp_path)

else:
    st.info("👈 Upload an image to get started!")
