from moviepy.editor import VideoFileClip
import streamlit as st
import os
import tempfile

def convert_to_mp3(uploaded_video):
    if uploaded_video is not None:
        # Define a path in your project directory to save the MP3 file
        output_path = os.path.join("/Users/amink/Desktop/pose/audio_download", "output_audio.mp3")

        # Save the uploaded video to a temporary file
        video_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        video_temp.write(uploaded_video.read())
        video_temp.close()

        # Extract audio and save as MP3
        with VideoFileClip(video_temp.name) as video_clip:
            video_clip.audio.write_audiofile(output_path)

        # Clean up temporary video file
        os.remove(video_temp.name)

        return output_path
    return None

# Streamlit interface
st.title("Video to MP3 Converter")

uploaded_video = st.file_uploader("Upload MP4 Video", type="mp4")
if uploaded_video is not None:
    audio_path = convert_to_mp3(uploaded_video)
    if audio_path:
        st.audio(audio_path)
        st.success("Audio extracted and converted to MP3. Saved at: " + audio_path)
