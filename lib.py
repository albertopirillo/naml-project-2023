"""
    Library file that contains functions shared across all notebooks in the project.
"""

from typing import List, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing


def pre_processing(df: pd.DataFrame) -> pd.DataFrame:
    """ Drop unnecessary columns and set the correct datatypes"""
    df = df.drop(['filename', 'length'], axis=1)
    dtypes = {col: 'np.float32' for col in df.columns}
    dtypes["label"] = "category"
    return df.astype(dtypes)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """ Normalize the numerical variables in the dataframe """
    df_num = df.select_dtypes(include='number')
    df_norm = (df_num - df_num.mean()) / df_num.std()
    df[df_norm.columns] = df_norm
    return df


def plot_accuracies(labels: List[str], accuracies: float, title: str) -> None:
    """ Generate a bar plot where the height is the accuracy """
    _, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=labels, y=accuracies, width=0.6, palette="winter", ax=ax)
    ax.set_title(title)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Classifier")
    ax.set_axisbelow(True)
    ax.grid(axis='x', visible=False)
    ax.grid(axis='y', color='gray')


def encode_labels(labels: np.ndarray[Any]) -> np.ndarray[np.float32]:
    """ One-hot encode the values of categorical variable into integer values """
    lb = preprocessing.LabelBinarizer()
    y = lb.fit_transform(labels)
    return y.astype(np.float32)


def balance_classes(data: np.ndarray[Any], labels: np.ndarray[Any], genre: str) \
        -> Tuple[np.ndarray[np.float32], np.ndarray[np.float32]]:
    """ Balance the elements of the two classes in a binary classification problem"""
    # Concatenate data and labels in a single matrix
    dataset = np.concatenate((data, labels[:, np.newaxis]), axis=1)

    # Split the data
    class_data = dataset[dataset[:, -1] == genre]
    other_data = dataset[dataset[:, -1] != genre]

    # Balance the classes
    idxs = np.random.choice(len(other_data), len(class_data))
    other_data = other_data[idxs]
    balanced_data = np.concatenate((class_data, other_data))

    # Separate inputs and labels
    X = balanced_data[:, :-1]
    y = balanced_data[:, -1]

    # Pre-process label vector
    y_new = y.copy()
    y_new[y != genre] = 0
    y_new[y == genre] = 1

    return X.astype(np.float32), y_new.astype(np.float32)
