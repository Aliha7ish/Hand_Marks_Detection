import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]


def extract_hand_landmarks(row):
    """
    Extract (x, y) hand landmarks from a dataframe row.

    Parameters
    ----------
    row : pd.Series
        One sample containing x1..x21, y1..y21, z1..z21

    Returns
    -------
    xs : np.ndarray
        X coordinates of 21 landmarks
    ys : np.ndarray
        Y coordinates of 21 landmarks
    """
    xs = np.array([row[f"x{i}"] for i in range(1, 22)])
    ys = np.array([row[f"y{i}"] for i in range(1, 22)])
    return xs, ys

def plot_hand_skeleton(xs, ys, ax):
    """
    Plot a hand skeleton using MediaPipe landmark connections.

    Parameters
    ----------
    xs : np.ndarray
        X coordinates of landmarks
    ys : np.ndarray
        Y coordinates of landmarks
    ax : matplotlib.axes.Axes
        Axis to plot on
    """
    ax.scatter(xs, ys, s=20)

    for start, end in HAND_CONNECTIONS:
        ax.plot(
            [xs[start], xs[end]],
            [ys[start], ys[end]],
            linewidth=1
        )

    ax.invert_yaxis()
    ax.axis("off")


def visualize_class_samples(df, label, n_samples=5):
    """
    Visualize hand landmark samples for a single class.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing hand landmarks and labels
    label : str
        Class label to visualize
    n_samples : int
        Number of samples to visualize
    """
    samples = df[df["label"] == label].sample(n_samples, random_state=42)

    fig, axes = plt.subplots(1, n_samples, figsize=(3 * n_samples, 3))
    fig.suptitle(f"Class: {label}", fontsize=14)

    for ax, (_, row) in zip(axes, samples.iterrows()):
        xs, ys = extract_hand_landmarks(row)
        plot_hand_skeleton(xs, ys, ax)

    plt.tight_layout()
    plt.show()


def visualize_samples_per_class(df, n_samples=5):
    """
    Visualize hand landmark samples for each class in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing hand landmarks and labels
    n_samples : int
        Number of samples per class
    """
    class_order = df["label"].value_counts().index

    for label in class_order:
        visualize_class_samples(df, label, n_samples)


def plot_class_distribution_bar(df, label_col="label", figsize=(10, 6)):
    """
    Plot the number of samples per class using a bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing class labels
    label_col : str
        Name of the label column
    figsize : tuple
        Figure size
    """
    class_counts = df[label_col].value_counts().reset_index()
    class_counts.columns = ["label", "count"]

    plt.figure(figsize=figsize)
    sns.barplot(
        data=class_counts,
        x="count",
        y="label"
    )

    plt.title("Samples per Class (Count)")
    plt.xlabel("Number of Samples")
    plt.ylabel("Class Label")
    plt.tight_layout()
    plt.show()


def plot_class_distribution_pie(df, label_col="label", figsize=(8, 8)):
    """
    Plot the percentage distribution of samples per class using a pie chart.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing class labels
    label_col : str
        Name of the label column
    figsize : tuple
        Figure size
    """
    class_counts = df[label_col].value_counts()
    percentages = class_counts / class_counts.sum() * 100

    plt.figure(figsize=figsize)
    plt.pie(
        percentages,
        labels=percentages.index,
        autopct="%.1f%%",
        startangle=90
    )

    plt.title("Samples per Class (Percentage)")
    plt.axis("equal")  # Ensures circular pie
    plt.tight_layout()
    plt.show()