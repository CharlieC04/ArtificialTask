import pandas as pd
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

# Function to augment descriptions in minority classes
def augment_descriptions(df, class_column, text_column, minority_classes, paraphraser, augment_factor=2):
    augmented_data = []
    for cls in minority_classes:
        subset = df[df[class_column] == cls]
        for _, row in subset.iterrows():
            original_text = row[text_column]
            # Generate paraphrased text
            for _ in range(augment_factor):
                paraphrased = paraphraser(f"paraphrase: {original_text}", max_length=128, num_return_sequences=1)
                augmented_text = paraphrased[0]['generated_text']
                augmented_data.append({**row, text_column: augmented_text})
    return pd.DataFrame(augmented_data)

# Function to generate embeddings
def generate_embeddings(texts, tokenizer, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # Use mean pooling to get sentence embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

def preprocess(data, training=False):
    class_distribution = data['country'].value_counts()
    minority_classes = class_distribution[class_distribution < 100].index.tolist()

    paraphraser = pipeline("text2text-generation", model="t5-small")

    # Augment the dataset
    augmented_data = augment_descriptions(
        data, class_column='country', text_column='description', 
        minority_classes=minority_classes, paraphraser=paraphraser, augment_factor=2
    )
    wine_data_augmented = pd.concat([data, augmented_data], ignore_index=True)

    # Split the augmented data into training and testing sets
    train_data, test_data = train_test_split(wine_data_augmented, test_size=0.2, random_state=42)
    X_train = train_data[['description', 'points', 'price', 'variety']]
    y_train = train_data['country']
    X_test = test_data[['description', 'points', 'price', 'variety']]
    y_test = test_data['country']

    # Load a Hugging Face model for embeddings
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Generate embeddings for training and testing descriptions
    train_embeddings = generate_embeddings(X_train['description'].tolist(), tokenizer, model)
    test_embeddings = generate_embeddings(X_test['description'].tolist(), tokenizer, model)

    # Preprocess numeric and categorical features
    numeric_features = ['points', 'price']
    categorical_features = ['variety']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    if training:
        # Save the preprocessor for future use
        with open("preprocessor.pkl", "wb") as f:
            pickle.dump(preprocessor, f)

    # Combine embeddings with preprocessed features
    X_train_combined = np.hstack([train_embeddings, X_train_processed.toarray()])
    X_test_combined = np.hstack([test_embeddings, X_test_processed.toarray()])

    return X_train_combined, X_test_combined, y_train, y_test


def loaders(data):
    X_train_combined, X_test_combined, y_train, y_test = preprocess(data)
    X_train_tensor = torch.tensor(X_train_combined, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.factorize()[0], dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_combined, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.factorize()[0], dtype=torch.long)
    # Prepare data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    return train_loader, test_loader