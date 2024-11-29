import os
import logging
import pandas as pd
import configparser
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import joblib

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Function to load config
def load_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config


# Ensure directories exist
def ensure_directory(path):
    os.makedirs(path, exist_ok=True)


# Load dataset
def load_data(file_path):
    logging.info(f"Loading dataset: {file_path}")
    return pd.read_csv(file_path, delimiter=';')


# Preprocess dataset
def preprocess_data(df):
    logging.info("Preprocessing dataset...")
    X = df.iloc[:, :-1].values  # Features: all except last column
    Y = df.iloc[:, -1].values  # Labels: last column (0 or 1)
    return X, Y


# Train model
def train_model(X_train, Y_train, model_used):
    model = None
    try:
        if model_used == '1':  # Random Forest
            logging.info("Training Random Forest Classifier...")
            model = RandomForestClassifier(random_state=42)
        elif model_used == '2':  # XGBoost
            logging.info("Training XGBoost Classifier...")
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        elif model_used == '3':  # Logistic Regression
            logging.info("Training Logistic Regression...")
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_used == '4':  # SVM
            logging.info("Training Support Vector Machine...")
            model = LinearSVC(random_state=42)
        elif model_used == '5':  # KNN
            logging.info("Training K-Nearest Neighbors...")
            model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, Y_train)
    except Exception as e:
        logging.error(f"Error during training: {e}")
    return model


# Evaluate model
def evaluate_model(model, X_test, Y_test, model_used, dataset_name, ds_type, vehicle):
    logging.info(f"Evaluating model for dataset: {dataset_name}...")
    Y_predicted = model.predict(X_test)

    # Confusion Matrix and Metrics
    cm = confusion_matrix(Y_test, Y_predicted)
    report = classification_report(Y_test, Y_predicted)
    accuracy = accuracy_score(Y_test, Y_predicted)

    # Save evaluation results
    if REPORT_ENABLE:
        ensure_directory(PATH_TO_MODEL_REPORT + ds_type + vehicle)
        model_mapping = {
            '1': 'Random Forest Classifier',
            '2': 'XGB Classifier',
            '3': 'Logistic Regression',
            '4': 'Support Vector Machine',
            '5': 'KNearest Neighbors'
        }
        model_name = model_mapping.get(model_used, 'UnknownModel')
        report_file_name = f"Evaluation_report_{dataset_name}.txt"

        with open(os.path.join(PATH_TO_MODEL_REPORT + ds_type + vehicle, report_file_name), "a") as file:
            file.write(f"{model_name} Report for {dataset_name}, Vehicle {vehicle}\n")
            file.write("--------------------------------------------------------------\n")
            file.write("Confusion Matrix:\n")
            file.write(str(cm) + "\n\n")
            file.write("--------------------------------------------------------------\n")
            file.write("Classification Report:\n")
            file.write(report + "\n\n")
            file.write("--------------------------------------------------------------\n")
            file.write(f"Accuracy: {accuracy:.4f}\n")
            file.write("--------------------------------------------------------------\n\n")

        logging.info(f"Evaluation results saved to {report_file_name}")

    # Print to console
    logging.info("\nConfusion Matrix:\n%s", cm)
    logging.info("\nClassification Report:\n%s", report)
    logging.info("\nAccuracy: %.4f", accuracy)

    # Plot Confusion Matrix
    if PLOT_CM:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Attack"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - {model_name} ({dataset_name})")
        plt.show()


# Save model
def save_model(model, model_used, dataset_name, ds_type, vehicle):
    ensure_directory('./Models/')
    model_mapping = {
        '1': 'RandomForestClassifier',
        '2': 'XGBClassifier',
        '3': 'LogisticRegression',
        '4': 'SupportVectorMachine',
        '5': 'KNearestNeighbors'
    }
    model_name = model_mapping.get(model_used, 'UnknownModel')
    model_file = f"{PATH_TO_MODEL_SAVE + ds_type + vehicle} + can_intrusion_model_{dataset_name}_{model_name}.pkl"
    joblib.dump(model, model_file)
    logging.info(f"Model saved to {model_file}")


# Main function
def main():
    # Ensure datasets folder exists
    if not os.path.exists(DATASETS_FOLDER):
        logging.error(f"Datasets folder not found: {DATASETS_FOLDER}")
        return

    dataset_types = [COMBINED, FUZZY, REPLAY]   # Attack types
    vehicles = [VEHICLE_A, VEHICLE_B, VEHICLE_C]    # Vehicles

    for ds_type in dataset_types:
        for vehicle in vehicles:
            # Iterate through datasets
            datasets = [f for f in os.listdir(DATASETS_FOLDER + ds_type + vehicle) if f.endswith('.csv')]
            if not datasets:
                logging.error("No datasets found in the folder.")
                return

            for dataset in datasets:
                dataset_path = os.path.join(DATASETS_FOLDER + ds_type + vehicle, dataset)
                dataset_name = os.path.splitext(dataset)[0]

                # Load and preprocess data
                df = load_data(dataset_path)
                X, Y = preprocess_data(df)
                logging.info(f"Dataset {dataset_name} successfully preprocessed!")

                # Split data into training and testing sets
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(1 - TRAINING_SIZE), random_state=42)
                logging.info(f"Data split for {dataset_name}: {TRAINING_SIZE * 100:.0f}% Training, {(1 - TRAINING_SIZE) * 100:.0f}% Testing")

                # Model selection
                model_selection = ''

                for model_selection in ['1', '2', '3', '4', '5']:
                    # Train the model
                    model = train_model(X_train, Y_train, model_selection)
                    if model is None:
                        logging.error(f"Model training failed for {dataset_name}. Skipping...")
                        continue
                    logging.info(f"Model successfully trained for {dataset_name}!")

                    # Evaluate the model
                    evaluate_model(model, X_test, Y_test, model_selection, dataset_name, ds_type, vehicle)

                    # Save the model
                    if MODEL_SAVE_ENABLE:
                        save_model(model, model_selection, dataset_name, ds_type, vehicle)
                        logging.info(f"Model successfully saved for {dataset_name}.")


if __name__ == "__main__":
    
    # Load configuration from the file
    config = load_config("Config/Configuration.txt")
    
    # Extract config
    DATASETS_FOLDER = config.get("Paths", "DATASETS_FOLDER")
    PATH_TO_MODEL_REPORT = config.get("Paths", "PATH_TO_MODEL_REPORT")
    PATH_TO_MODEL_SAVE = config.get("Paths", "PATH_TO_MODEL_SAVE")

    COMBINED = config.get("DatasetTypes", "COMBINED")
    FUZZY = config.get("DatasetTypes", "FUZZY")
    REPLAY = config.get("DatasetTypes", "REPLAY")

    VEHICLE_A = config.get("Vehicles", "VEHICLE_A")
    VEHICLE_B = config.get("Vehicles", "VEHICLE_B")
    VEHICLE_C = config.get("Vehicles", "VEHICLE_C")

    TRAINING_SIZE = float(config.get("ModelParameters", "TRAINING_SIZE"))

    MODEL_SAVE_ENABLE = config.getboolean("Macros", "MODEL_SAVE_ENABLE")
    REPORT_ENABLE = config.getboolean("Macros", "REPORT_ENABLE")
    PLOT_CM = config.getboolean("Macros", "PLOT_CM")

    main()
