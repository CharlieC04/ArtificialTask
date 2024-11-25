from preprocess import preprocess, loaders
import argparse
import pickle
import torch
import pandas as pd
from tune import tune
from train import train
from eval import eval
from predict import predict
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wine Quality Model Script")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Tune subcommand
    subparsers.add_parser("tune", help="Tune the model and save the best parameters")

    # Train subcommand with optional --tune-first flag
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "-th", "--tune-first", action="store_true", help="Tune the model first before training"
    )

    # Eval subcommand
    eval_parser = subparsers.add_parser("predict", help="Use the model to predict values for a specified file")
    eval_parser.add_argument("file", type=str, help="File to evaluate the model on")

    # Parse the arguments
    args = parser.parse_args()
    if args.command not in ["tune", "train", "predict"]:
        parser.print_help()
        sys.exit()

    if args.command != "predict":
        file_path = 'wine_quality_1000.csv'  # Replace with your file path in Colab
        wine_data = pd.read_csv(file_path).drop(columns=['Unnamed: 0']).dropna()
        X_train_combined, X_test_combined, y_train, y_test = preprocess(wine_data, args.command == "train")
        train_loader, test_loader = loaders(wine_data)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    

    # Handle commands
    if args.command == "tune":
        params = tune(X_train_combined, y_train, train_loader, test_loader, device)
        with open("best_params.pkl", "wb") as f:
            pickle.dump(params, f)
        print("Best parameters saved to best_params.pkl")

    elif args.command == "train":
        if args.tune_first:
            best_params = tune(X_train_combined, y_train, train_loader, test_loader, device)
            with open("best_params.pkl", "wb") as f:
                pickle.dump(best_params, f)
            print("Best parameters saved to best_params.pkl")
        else:
            try:
                with open("best_params.pkl", "rb") as f:
                    best_params = pickle.load(f)
                print(f"Loaded best parameters: {best_params}")
            except FileNotFoundError:
                print("No best parameters found. Training with default settings.")
                best_params = None
        if best_params is not None:
            model, class_labels = train(best_params, X_train_combined, y_train, train_loader, device)
            with open("trained_model.pkl", "wb") as f:
                pickle.dump({'trained_model': model, 'class_labels': class_labels}, f)
            print("Model and class labels saved to 'trained_model.pkl'.")
            eval(model, y_train, test_loader, device)
            

    elif args.command == "predict":
        try:
            with open("trained_model.pkl", "rb") as f:
                model = pickle.load(f)
                print(f"Loaded model: {model}")
                predict(model, args.file, device)

        except FileNotFoundError:
            print("No trained model found. Please train the model first.")