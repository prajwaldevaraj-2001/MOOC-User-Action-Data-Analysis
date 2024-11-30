
import pandas as pd
import os
import requests

# Corrected download URL for the MOOC dataset
def download_dataset():
    # Provide the correct URL to the dataset (you should manually download it from the SNAP page if URL doesn't work)
    url = "https://snap.stanford.edu/data/act-mooc.txt"  # Replace with the correct file if needed
    dataset_path = "data/act-mooc.tsv"  # This is where you expect the dataset to be saved

    if not os.path.exists("data"):
        os.makedirs("data")

    # Check if file already exists
    if not os.path.exists(dataset_path):
        print("Downloading the dataset...")

        try:
            # Use requests to download the dataset
            response = requests.get(url)
            response.raise_for_status()  # Check if the request was successful
            with open(dataset_path, "wb") as file:
                file.write(response.content)
            print(f"Dataset downloaded and saved to {dataset_path}.")
        except requests.exceptions.HTTPError as err:
            print(f"HTTP Error: {err}")
            print("Failed to download the dataset.")
            return None
    else:
        print("Dataset already exists.")
    
    return dataset_path

# Load and process the dataset
def process_dataset(filepath):
    print("Loading dataset...")
    
    # Ensure the path points to a valid file, not a directory
    if os.path.isdir(filepath):
        print(f"Error: {filepath} is a directory, not a file!")
        return None

    try:
        data = pd.read_csv(filepath, sep="	", header=None)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    # Rename columns for clarity
    data.columns = ["user_id", "target_id", "action", "timestamp"]
    
    # Convert timestamp to readable format
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s', origin='unix')
    
    print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns.")
    print("\nSample Data:")
    print(data.head())
    
    return data

# Export data for Tableau
def export_for_tableau(data):
    export_path = "data/processed_mooc_data.csv"
    print(f"Exporting data to {export_path} for Tableau analysis...")
    data.to_csv(export_path, index=False)
    print("Data exported successfully.")

def main():
    # Step 1: Download the dataset
    dataset_path = download_dataset()
    
    if dataset_path is None:
        print("Exiting due to download failure.")
        return
    
    # Step 2: Process the dataset
    data = process_dataset(dataset_path)
    
    if data is None:
        print("Exiting due to dataset processing failure.")
        return
    
    # Step 3: Export the dataset for Tableau
    export_for_tableau(data)

if __name__ == "__main__":
    main()
