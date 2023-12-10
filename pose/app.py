import streamlit as st
st.set_page_config(page_title='Pain Detection App', layout='wide')
from utils import make_circular
from face_mesh_processor import process_video
import pandas as pd
import numpy as np
import emotion_graph

# Set page config for wide mode
#st.set_page_config(page_title='Pain Detection App', layout='wide')


# Custom CSS for styling
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
        }
        h1 {
            text-align: center;
            color: #0e76a8;
            font-size: 3rem; /* Larger header size */
            margin-bottom: 1rem; /* Space below the header */
        }
        .stFileUploader, .stSlider, .stImage {
            margin: 1rem 0; /* Space around file uploaders, slider, and images */
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1>Pain Detection App</h1>", unsafe_allow_html=True)
''''''
# Images and video uploaders
'''
col1, col2 = st.columns(2, gap="large")
with col1:
    patient_image = make_circular("/Users/amink/Desktop/pose/static/patient.png")
    st.image(patient_image, caption="Patient", width=120)
    uploaded_patient_video = st.file_uploader("Upload Patient Video", type=["mp4", "mov", "avi", "mkv"], key="patient_video_uploader")
    if uploaded_patient_video is not None:
        process_video(uploaded_patient_video, col1)

with col2:
    clinician_image = make_circular("/Users/amink/Desktop/pose/static/doctor.png")
    st.image(clinician_image, caption="Clinician", width=120)
    uploaded_clinician_video = st.file_uploader("Upload Clinician Video", type=["mp4", "mov", "avi", "mkv"], key="clinician_video_uploader")
    if uploaded_clinician_video is not None:
        process_video(uploaded_clinician_video, col2)

# Pain Meter
st.subheader("Pain Meter")
pain_level = st.slider("", min_value=0, max_value=100, value=50, step=1, key="pain_meter")
st.text("Low Pain" if pain_level < 35 else "Moderate Pain" if pain_level < 70 else "High Pain")
'''

def display_emotion_graph():
    df = emotion_graph.load_data('/Users/amink/Desktop/pose/face.csv')
    if df.empty:
        st.write("No data available to display the graph.")
    else:
        face_0_df = emotion_graph.filter_data(df, 'face_0')
        emotions_to_plot = ["Admiration", "Adoration", "Anger", "Anxiety", "Awe", "Boredom", "Calmness", "Confusion", "Contempt", "Desire"]
        fig = emotion_graph.create_emotion_figure(face_0_df, emotions_to_plot)
        st.plotly_chart(fig)
