import pandas as pd

def load_data(path):
    """
    Load the train, dev, test datasets from the specified path 
    and return the features and labels for each set.
    
    path: str, the directory path where the CSV files are located.

    Returns:
    X_train, y_train, X_dev, y_dev, X_test, y_test: DataFrames and Series for features and labels.

    """
    
    # load train, dev, test sets
    train_df = pd.read_csv(f"{path}/hand_landmarks_train.csv")
    dev_df = pd.read_csv(f"{path}/hand_landmarks_dev.csv")
    test_df = pd.read_csv(f"{path}/hand_landmarks_test.csv")

    # unback the features and labels
    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    
    X_dev = dev_df.drop(columns=["label"])
    y_dev = dev_df["label"]

    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    return X_train, y_train, X_dev, y_dev, X_test, y_test
