import streamlit as st
from transformers import pipeline
import speech_recognition as sr
import soundfile as sf

# Initialize the question-answering pipeline
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Function to take microphone input and convert it to text
def get_audio_input():
    recognizer = sr.Recognizer()
    with sr.AudioFile('temp.wav') as source:
        st.info("Listening...")
        audio = recognizer.record(source)
        try:
            query = recognizer.recognize_google(audio)
            st.success(f"You said: {query}")
            return query
        except sr.UnknownValueError:
            st.error("Google Speech Recognition could not understand the audio.")
        except sr.RequestError:
            st.error("Could not request results from Google Speech Recognition service.")
    return None

st.title("QA Chatbot with Keyboard and Microphone Input")

# Choose input method
input_method = st.radio("Choose your input method:", ("Keyboard", "Microphone"))

if input_method == "Keyboard":
    question = st.text_input("Enter your question:")
elif input_method == "Microphone":
    # You can record a WAV file using an external recorder and load it
    # Use 'temp.wav' file as a placeholder
    question = get_audio_input()

if question:
    with st.spinner("Finding the answer..."):
        result = qa_pipeline(question=question, context="Your context or document goes here.")
        st.write(f"**Answer:** {result['answer']}")
