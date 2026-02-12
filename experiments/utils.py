import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

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

def train_model(model, X_train, y_train):
    """
    Train the given model using the provided training data.
    
    :param model: sklearn model instance, the machine learning model to be trained.
    :param X_train: df.DataFrame, the training features.
    :param y_train: df.Series, The training labels

    return: the trained model. 
    """

    model.fit(X_train, y_train)

    return model
    
def evaluate_model(model, X_dev, y_dev):
    """
    Evaluate the performance of the trained model on the development set and return the evaluation metrics.
    
    :param model: trained model instance, the machine learning model to be evaluated.
    :param X_dev: df.DataFrame, the development features.
    :param y_dev: df.Series, The development labels.

    Returns:
    metrics: dict, dictionary containing the performance metrics as accuracy, precision, recall, and F1-score. 

    """

    y_pred = model.predict(X_dev)

    accuracy = accuracy_score(y_dev, y_pred)
    precision = precision_score(y_dev, y_pred)
    recall = recall_score(y_dev, y_pred)
    f1 = f1_score(y_dev, y_pred)
    cm = confusion_matrix(y_dev, y_pred)

    # strore them into metrics dict
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }

    return metrics


