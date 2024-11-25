# Wine Quality Model Script

This repository contains a Python script for tuning, training, and evaluating a machine learning model to predict the country of origin of wines based on their description and other features.

## Features

The script provides the following functionalities:
1. **Tuning**: Tune hyperparameters of the model and save the best parameters.
2. **Training**: Train the model using the best parameters or default settings.
3. **Prediction**: Predict the labels of an unseen dataset (for use in production).

## Prerequisites

1. Python 3.7 or later.
2. Required Python libraries:
   - `torch`
   - `transformers`
   - `scikit-learn`
   - `optuna`
   - `pandas`
   - `numpy`
   - `argparse`

Install the required dependencies using:
```bash
pip install -r requirements.txt
```
3. Enter your huggingface access token to use the LLMs, for example via the following command after installing the required module:
```bash
huggingface-cli login
```
## Usage

Run the script using the command-line interface to perform different tasks:

### 1. **Tuning the Model**
To tune the model and save the best hyperparameters:
```bash
python main.py tune
```

### 2. **Training the Model**
To train the model:
- Without tuning:
  ```bash
  python main.py train
  ```
- With tuning first:
  ```bash
  python main.py train -th
  ```

The trained model will be saved as `trained_model.pkl`.

### 3. **Prediction**
To predict the labels of unseen data (CSV file without country labels):
```bash
python main.py predict <path_to_csv>
```
Replace `<path_to_csv>` with the file path of the dataset to evaluate.

### Example:
```bash
python main.py predict new_wine_data.csv
```

## Input Requirements

- **Training Dataset**:
  The script expects a CSV file with the following columns:
  - `country`: The target label (only required during training).
  - `description`: Free-text description of the wine.
  - `points`: Numeric review score of the wine (1-100).
  - `price`: Price of the wine (numeric).
  - `variety`: Grape variety used.

- **Evaluation Dataset**:
  For prediction, the dataset must contain all the columns listed above **except** for the `country` column.

## Outputs

1. **Tuned Parameters**:
   - Saved as `best_params.pkl`.

2. **Trained Model**:
   - Saved as `trained_model.pkl`.

3. **Predicted Results**:
   - The `predict` command appends a new column, `predicted_country`, to the input dataset and saves it as `predicted_wine_data.csv`.

## File Structure

```
.
├── main.py                  # Main script for tuning, training, and evaluation
├── predict.py               # Standalone script for predicting on unlabeled data
├── requirements.txt         # Required dependencies
├── trained_model.pkl        # Trained model (generated after training)
├── best_params.pkl          # Best hyperparameters (generated after tuning)
├── predicted_wine_data.csv  # Output file after evaluation
```