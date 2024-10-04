import streamlit as st
from model_loader import (
    load_huggingface_model, load_openai_model, load_azure_openai_model,
    load_custom_model, load_fasttext_model, load_spacy_model, train_huggingface_model
)
from token_counter import count_tokens
import torch

HUGGINGFACE_TOKEN = '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'
OPENAI_API_KEY = 'your_openai_api_key'
AZURE_OPENAI_API_KEY = 'your_azure_openai_api_key'
AZURE_OPENAI_ENDPOINT = 'your_azure_openai_endpoint'

@st.cache_resource
def load_models():
    models = {}
    models['albert/albert-base-v2'] = load_huggingface_model("albert-base-v2", HUGGINGFACE_TOKEN)
    models['vineetsharma/customer-support-intent-albert'] = load_huggingface_model("vineetsharma/customer-support-intent-albert", HUGGINGFACE_TOKEN)
    # Add more models as needed
    return models

models = load_models()

st.title("Intent Classification App")

model_choice = st.selectbox("Choose a model", list(models.keys()) + ['OpenAI', 'Azure OpenAI', 'Custom Model', 'FastText', 'spaCy'])
input_text = st.text_input("Enter text")

if st.button("Classify Intent"):
    if model_choice in models:
        tokenizer, model = models[model_choice]
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        predicted_label = model.config.id2label[predicted_class]
        result = predicted_label
    elif model_choice == 'OpenAI':
        model = load_openai_model(OPENAI_API_KEY)
        # Add logic to classify intent using OpenAI model
        result = "OpenAI result"
    elif model_choice == 'Azure OpenAI':
        client = load_azure_openai_model(AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT)
        # Example logic to classify intent using Azure OpenAI model
        response = client.Completions.create(prompt=input_text, max_tokens=50)
        result = response.choices[0].text.strip()
    elif model_choice == 'Custom Model':
        model = load_custom_model('path_to_custom_model')
        # Add logic to classify intent using custom model
        result = "Custom Model result"
    elif model_choice == 'FastText':
        model = load_fasttext_model('path_to_fasttext_model')
        result = model.predict(input_text)[0][0]
    elif model_choice == 'spaCy':
        nlp = load_spacy_model('en_core_web_sm')
        doc = nlp(input_text)
        result = doc.cats  # Assuming a text categorization model

    token_count = count_tokens(input_text)
    st.write("Intent:", result)
    st.write("Token Count:", token_count)

st.header("Train a Model")
train_model_choice = st.selectbox("Choose a model to train", list(models.keys()))
train_texts = st.text_area("Enter training texts (one per line)")
train_labels = st.text_area("Enter training labels (one per line)")
val_texts = st.text_area("Enter validation texts (one per line)")
val_labels = st.text_area("Enter validation labels (one per line)")

if st.button("Train Model"):
    train_data = {'text': train_texts.split('\n'), 'label': [int(label) for label in train_labels.split('\n')]}
    val_data = {'text': val_texts.split('\n'), 'label': [int(label) for label in val_labels.split('\n')]}
    model, tokenizer = train_huggingface_model(train_model_choice, train_data, val_data, HUGGINGFACE_TOKEN)
    st.write(f"Model {train_model_choice} trained and saved.")