import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
def load_data(filepath):
    print("Loading dataset...")
    data = pd.read_csv(filepath, sep="\t")
    print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns.")
    return data

# Preprocess the data
def preprocess_data(data):
    print("Preprocessing data...")
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
    data['action'] = data['action'].astype(int)
    print("Data preprocessing complete.")
    return data

# Perform exploratory data analysis
def perform_eda(data):
    print("\nExploratory Data Analysis:")
    print(data.describe())
    print("\nTop 5 rows of the dataset:")
    print(data.head())

    # Plot user actions over time
    print("\nGenerating plot for user actions over time...")
    plt.figure(figsize=(10, 5))
    plt.plot(data['timestamp'], data['action'], alpha=0.5, label='User Actions')
    plt.xlabel("Time")
    plt.ylabel("Actions")
    plt.title("User Actions Over Time")
    plt.legend()
    plt.grid()
    plt.savefig('user_actions_over_time.png')
    print("Plot saved as 'user_actions_over_time.png'.")

# Build and evaluate a simple predictive model
def build_model(data):
    print("\nBuilding and evaluating a simple predictive model...")
    
    # Prepare features and labels
    data['label'] = data['action'] % 2  # Example: classify based on action being even/odd
    X = data[['action']]  # Feature
    y = data['label']     # Target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def main():
    filepath = "data/act-mooc.tsv"
    
    # Load data
    data = load_data(filepath)
    
    # Preprocess data
    data = preprocess_data(data)
    
    # Perform exploratory data analysis
    perform_eda(data)
    
    # Build and evaluate a simple predictive model
    build_model(data)

if __name__ == "__main__":
    main()
