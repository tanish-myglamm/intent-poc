import pandas as pd
import torch
from transformers import AlbertForSequenceClassification, AlbertTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
from data_preparation import prepare_data
import streamlit as st
import os

# Hugging Face API token to access pretrained models
HUGGINGFACE_TOKEN = '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'

# Step 1: Prepare the Data
# Load training and validation data using the `prepare_data` function from data_preparation module
train_df, val_df = prepare_data()

# Convert the Pandas DataFrame to Hugging Face Dataset format
# This is required for compatibility with the Hugging Face Trainer API
dataset_train = Dataset.from_pandas(train_df)
dataset_val = Dataset.from_pandas(val_df)

# Step 2: Load Model and Tokenizer
# Load the tokenizer and model from Hugging Face. The tokenizer is used to convert text into input tensors
model_name = "albert-base-v2"
tokenizer = AlbertTokenizer.from_pretrained(model_name)
model = AlbertForSequenceClassification.from_pretrained(model_name, num_labels=5)

# Step 3: Tokenize the Data
# Define a function to tokenize the dataset. The tokenization step is essential for converting text to a format
# the model can understand (input IDs and attention masks)
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

# Apply tokenization to the training and validation datasets
# The `map` function applies the tokenization function to each example in the dataset
# `batched=True` ensures that multiple examples are processed at the same time, which can speed up tokenization
tokenized_train = dataset_train.map(tokenize_function, batched=True)
tokenized_val = dataset_val.map(tokenize_function, batched=True)

# Step 4: Define Training Arguments
# Define the parameters for training. These settings control aspects of training such as learning rate, batch size,
# number of epochs, and where to store output logs and model checkpoints
training_args = TrainingArguments(
    output_dir="./results",  # Directory to save model checkpoints and other results
    evaluation_strategy="epoch",  # Evaluate the model at the end of each epoch
    learning_rate=2e-5,  # Learning rate for the optimizer
    per_device_train_batch_size=8,  # Batch size for training
    per_device_eval_batch_size=8,  # Batch size for evaluation
    num_train_epochs=3,  # Number of training epochs
    weight_decay=0.01,  # Weight decay to prevent overfitting
    logging_dir='./logs',  # Directory to save training logs
    logging_steps=10,  # Log training metrics every 10 steps
)

# Step 5: Initialize Trainer
# The Trainer class in Hugging Face provides an easy way to train and evaluate the model
trainer = Trainer(
    model=model,  # The model to train
    args=training_args,  # Training arguments defined earlier
    train_dataset=tokenized_train,  # Tokenized training dataset
    eval_dataset=tokenized_val,  # Tokenized validation dataset
)

# Step 6: Train the Model
# Train the model using the specified training arguments and datasets
trainer.train()

# Step 7: Save the Model
# Save the fine-tuned model and tokenizer to the specified directory for future use
trainer.save_model("./trained_albert_intent_classifier")
tokenizer.save_pretrained("./trained_albert_intent_classifier")

# The above code will fine-tune the Albert model for a use case that classifies a given query into one of several predefined intents.
# Steps involved:
# 1. Data preparation and tokenization.
# 2. Use Hugging Face's Trainer API for model training and evaluation.
# 3. Save the trained model for further use.

# Note: Make sure to adjust the dataset and intent labels as per your use case.

# Step 8: Streamlit App to Use the Trained Model
# Create a simple Streamlit application to test the trained model

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
            predicted_label = torch.argmax(outputs.logits, dim=1).item()
            # Display the result
            st.write(f"Predicted Intent Label: {predicted_label}")
        else:
            st.write("Please enter a query to classify.")
else:
    st.write("Model not found. Please ensure the model is trained and saved at the specified path.")