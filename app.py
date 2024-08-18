import streamlit as st
from transformers import pipeline

# Load the QA pipeline with BERT-Large model
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Initialize context and conversation history in Streamlit session state
if 'context' not in st.session_state:
    st.session_state.context = ""
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'show_history' not in st.session_state:
    st.session_state.show_history = True

st.title("QA Chatbot with BERT-Large")

# Context input
context_input = st.text_area("Set the context:", placeholder="Enter context or leave blank to keep previous context.")
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
question_input = st.text_input("Ask a question:")
if st.button("Submit"):
    if not st.session_state.context:
        st.write("Please provide a context first.")
    else:
        result = qa_pipeline(question=question_input, context=st.session_state.context)
        st.session_state.conversation.append((question_input, result['answer']))
        st.write(f"Answer: {result['answer']}")

        # Option to ask another question
        if st.button("Ask Another Question"):
            question_input = ""  # Clear the question input for the next question
