import streamlit as st
import numpy as np
import cv2
import tempfile
from tensorflow.keras.models import load_model

# ================== CONFIG ==================
IMG_SIZE = 299  # change if your model uses 299 etc.
FRAME_SKIP = 10  # process every 10th frame (speed vs accuracy)

# ================== LOAD MODEL ==================
@st.cache_resource
def load_my_model():
    return load_model("best_deepfake_model.keras")

model = load_my_model()

# ================== PREPROCESS ==================
def preprocess_frame(frame):
    frame = cv2.resize(frame, (299,299))
    frame = frame / 255.0
    return frame

# ================== VIDEO PREDICTION ==================
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)

    predictions = []
    frame_count = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    progress_bar = st.progress(0)
    status_text = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every Nth frame
        if frame_count % FRAME_SKIP == 0:
            processed = preprocess_frame(frame)
            processed = np.expand_dims(processed, axis=0)

            pred = model.predict(processed, verbose=0)[0][0]
            predictions.append(pred)

        # Update progress UI
        progress = min(frame_count / total_frames, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")

        frame_count += 1

    cap.release()
    progress_bar.empty()
    status_text.empty()

    if len(predictions) == 0:
        return 0

    return float(np.mean(predictions))

# ================== UI ==================
st.set_page_config(page_title="Deepfake Detector", layout="centered")

st.title("🎥 Deepfake Video Detection")
st.write("Upload a video to detect whether it is **Real or Fake**")

uploaded_video = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov"]
)

if uploaded_video is not None:
    st.video(uploaded_video)

    # Save video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    st.info("Click below to analyze the video")

    if st.button("🔍 Analyze Video"):
        with st.spinner("Analyzing video... This may take some time ⏳"):
            result = predict_video(tfile.name)

        st.subheader("Result:")

        # ================== OUTPUT ==================
        if result > 0.5:
            st.error(f"⚠️ Fake Video (Confidence: {result:.2f})")
        else:
            st.success(f"✅ Real Video (Confidence: {1 - result:.2f})")

        st.progress(float(result))