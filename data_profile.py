from pathlib import Path
from typing import List
import pandas as pd
import numpy as np


##############################################
# Example(s). Read the comments in the following method(s)
##############################################
def pandas_profile(df: pd.DataFrame, result_html: str = 'report.html'):
    """
    This method will be responsible to extract a pandas profiling report from the dataset.
    Do not change this method, but run it and look through the html report it generated.
    Always be sure to investigate the profile of your dataset (max, min, missing values, number of 0, etc).
    """
    from pandas_profiling import ProfileReport

    profile = ProfileReport(df, title="Pandas Profiling Report")
    if result_html is not None:
        profile.to_file(result_html)
    return profile.to_json()


##############################################
# Implement all the below methods
# All methods should be dataset-independent, using only the methods done in the assignment
# so far and pandas/numpy/sklearn for the operations
##############################################
def get_column_max(df: pd.DataFrame, column_name: str) -> float:
    # Returning maximum value of a column
    return df[column_name].max()


def get_column_min(df: pd.DataFrame, column_name: str) -> float:
    # Returning minimum value of a column
    return df[column_name].min()


def get_column_mean(df: pd.DataFrame, column_name: str) -> float:
    # Returning mean of a column
    return df[column_name].mean()


def get_column_count_of_nan(df: pd.DataFrame, column_name: str) -> float:
    """
    This is also known as the number of 'missing values'
    """
    # Finding the number of null values and totalling it.
    return df[column_name].isnull().sum()


def get_column_number_of_duplicates(df: pd.DataFrame, column_name: str) -> float:
    # Calculating the duplicates
    return df[column_name].duplicated()


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    # Returning the list of numeric columns
    return df.select_dtypes(include=np.number).columns.tolist()


def get_binary_columns(df: pd.DataFrame) -> List[str]:
    # Returning the list of binary columns
    return df.select_dtypes(include=np.bool_).columns.tolist()


def get_text_categorical_columns(df: pd.DataFrame) -> List[str]:
    # Returning the list of textual columns
    return df.select_dtypes(exclude=np.number).columns.tolist()


def get_correlation_between_columns(df: pd.DataFrame, col1: str, col2: str) -> float:
    """
    Calculate and return the pearson correlation between two columns
    """
    # Calculating the correlation
    column1 = df[col1]
    column2 = df[col2]
    return column1.corr(column2)

