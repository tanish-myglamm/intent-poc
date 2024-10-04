import os
import torch
import streamlit as st
from transformers import AlbertTokenizer, AlbertForSequenceClassification

st.title("Intent Classification Model Tester")

# Load the trained model and tokenizer
model_path = "./trained_albert_intent_classifier"

# Check if the model and tokenizer files exist before loading them
if os.path.exists(model_path):
    tokenizer = AlbertTokenizer.from_pretrained(model_path)
    model = AlbertForSequenceClassification.from_pretrained(model_path)

    # User input
    input_text = st.text_input("Enter a query for intent classification:")

    if st.button("Classify Intent"):
        if input_text:
            # Tokenize the input text
            inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True)
            # Perform inference
            outputs = model(**inputs)
            # Get the predicted label
            predicted_class = torch.argmax(outputs.logits, dim=1).item()
            predicted_label = model.config.id2label[predicted_class]
            # Display the result
            st.write(f"Predicted Intent Label: {predicted_label}")
        else:
            st.write("Please enter a query to classify.")
else:
    st.write("Model not found. Please ensure the model is trained and saved at the specified path.")