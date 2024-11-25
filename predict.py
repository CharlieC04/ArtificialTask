import pandas as pd
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

def generate_embeddings(texts, tokenizer, embedding_model, device):
    """
    Generate embeddings for a list of texts using a Hugging Face model.
    :param texts: List of text descriptions.
    :param tokenizer: Hugging Face tokenizer.
    :param embedding_model: Hugging Face embedding model.
    :param device: Device for computation (CPU or GPU).
    :return: Numpy array of embeddings.
    """
    embedding_model.to(device)
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

def predict(model, file_path, device):
    # Load the Hugging Face embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embedding_model = AutoModel.from_pretrained(model_name)
    embedding_model.to(device)
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
    
    # Load the preprocessor
    try:
        with open("preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)
    except FileNotFoundError:
        print("No preprocessor found. Please ensure preprocessing is saved during training.")
        return
    
    # Validate columns
    required_columns = ['description', 'points', 'price', 'variety']
    if not all(col in data.columns for col in required_columns):
        print(f"Input CSV must contain columns: {', '.join(required_columns)}")
        return
    
    # Generate embeddings for descriptions
    descriptions = data['description'].tolist()
    embeddings = generate_embeddings(descriptions, tokenizer, embedding_model, device)

    # Preprocess numeric and categorical features
    features = data[['points', 'price', 'variety']]
    processed_features = preprocessor.transform(features)

    # Combine embeddings with preprocessed features
    combined_features = np.hstack([embeddings, processed_features.toarray()])

    # Convert combined features to a PyTorch tensor
    combined_features_tensor = torch.tensor(combined_features, dtype=torch.float32).to(device)

    # Predict with the trained model
    class_labels = model['class_labels']
    model = model['trained_model']
    model.eval()
    with torch.no_grad():
        outputs = model(combined_features_tensor)
        _, predicted_indices = torch.max(outputs, 1)

    # Map numerical predictions back to class labels
    predicted_labels = [class_labels[idx] for idx in predicted_indices.cpu().numpy()]

    # Add predictions to the data and save to a new file
    data['predicted_country'] = predicted_labels
    output_file = "predicted_wine_data.csv"
    data.to_csv(output_file, index=False)
    print(f"Predicted country labels saved to {output_file}")




