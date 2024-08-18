import streamlit as st
from transformers import pipeline
import os

# Load the QA pipeline with BERT-Language Large model
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Initialize context and conversation history in Streamlit session state
if 'context' not in st.session_state:
    st.session_state.context = ""
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'show_history' not in st.session_state:
    st.session_state.show_history = True
if 'question_input' not in st.session_state:
    st.session_state.question_input = ""

st.set_page_config(page_title="QA Chatbot with BERT-Large")

# Add custom CSS for background image and centralized text
image_path = "APP_PHOTO.jpg"  # Adjust the path as needed

# Ensure the image path is valid and accessible
if os.path.isfile(image_path):
    st.markdown(
        f"""
        <style>
        .reportview-container {{
            background: url("{image_path}") no-repeat center center fixed;
            background-size: cover;
            height: 100vh;  /* Ensure the background covers the entire viewport height */
            color: white;  /* Set default text color to white */
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }}
        .sidebar .sidebar-content {{
            background: rgba(0, 0, 0, 0.3); /* Optional: to give some transparency to the sidebar */
            color: white;  /* Set sidebar text color to white */
        }}
        .stTextInput input {{
            color: black; /* Set input text color to black */
            background-color: rgba(255, 255, 255, 0.8); /* Slightly transparent white background */
            border: 1px solid white; /* White border */
            text-align: center; /* Center text in input box */
        }}
        .stTextInput input:focus {{
            border-color: white; /* White border on focus */
        }}
        .stTextArea textarea {{
            color: black; /* Set textarea text color to black */
            background-color: rgba(255, 255, 255, 0.8); /* Slightly transparent white background */
            border: 1px solid white; /* White border */
            text-align: center; /* Center text in textarea */
        }}
        .stTextArea textarea:focus {{
            border-color: white; /* White border on focus */
        }}
        .stMarkdown {{
            text-align: center; /* Center text in markdown elements */
        }}
        .stApp {{
        background-image: url('https://static.vecteezy.com/system/resources/previews/007/353/692/non_2x/blue-crumpled-paper-question-mark-on-yellow-background-photo.jpg');
        background-color: rgba(0, 0, 0, 0.7);
        }}
        .stButton {{
            display: flex;
            justify-content: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.write("Background image file not found. Please check the path.")

st.title("QA Chatbot with BERT-Large")

# Context input
context_input = st.text_area("Set the context:", placeholder="Enter a context/paragraph(any language), then click *Update context* .")
if st.button("Update Context"):
    st.session_state.context = context_input
    st.write("Context updated.")

# Toggle for showing/hiding conversation history
if st.button("Toggle Conversation History"):
    st.session_state.show_history = not st.session_state.show_history

# Display conversation history
if st.session_state.show_history:
    st.write("Conversation:")
    for i, (question, answer) in enumerate(st.session_state.conversation):
        st.write(f"Q{i+1}: {question}")
        st.write(f"A{i+1}: {answer}")

# Question input
st.session_state.question_input = st.text_input("Ask a question:", value=st.session_state.question_input)
if st.button("Submit"):
    if not st.session_state.context:
        st.write("Please provide a context first.")
    else:
        result = qa_pipeline(question=st.session_state.question_input, context=st.session_state.context)
        st.session_state.conversation.append((st.session_state.question_input, result['answer']))
        st.write(f"Answer: {result['answer']}")

        # Option to ask another question
        if st.button("Ask Another Question"):
            st.session_state.question_input = ""  # Clear the question input for the next question
            st.experimental_rerun()  # Rerun the script to clear the input box
