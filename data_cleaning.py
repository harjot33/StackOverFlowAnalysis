import collections
from pathlib import Path
from typing import Union, Optional
from enum import Enum

import pandas as pd
import numpy as np

from data_profile import *
# from assignment1.a_load_file import read_dataset


##############################################
# Example(s). Read the comments in the following method(s)
##############################################
class WrongValueNumericRule(Enum):
    """
    You'll use these enumeration possibilities in your implemented methods below
    """
    MUST_BE_POSITIVE = 0
    MUST_BE_NEGATIVE = 1
    MUST_BE_GREATER_THAN = 2
    MUST_BE_LESS_THAN = 3


class DistanceMetric(Enum):
    """
    You'll use these enumeration possibilities in your implemented methods below
    """
    EUCLIDEAN = 0
    MANHATTAN = 1


##############################################
# Implement all the below methods
# All methods should be dataset-independent, using only the methods done in the assignment
# so far and pandas/numpy/sklearn for the operations
##############################################
def fix_numeric_wrong_values(df: pd.DataFrame,
                             column: str,
                             must_be_rule: WrongValueNumericRule,
                             must_be_rule_optional_parameter: Optional[float] = None) -> pd.DataFrame:
    """
    This method should fix the wrong_values depending on the logic you think best to do and using the rule passed by parameter.
    Remember that wrong values are values that are in the dataset, but are wrongly inputted (for example, a negative age).
    Here you should only fix them (and not find them) by replacing them with np.nan ("not a number", or also called "missing value")
    :param df: Dataset
    :param column: the column to be investigated and fixed
    :param must_be_rule: one of WrongValueNumericRule identifying what rule should be followed to flag a value as a wrong value
    :param must_be_rule_optional_parameter: optional parameter for the "greater than" or "less than" cases
    :return: The dataset with fixed column
    """
    if column in get_numeric_columns(df):
        if must_be_rule is WrongValueNumericRule.MUST_BE_POSITIVE:  # Implementing rule for must be positive rule
            mask = (df[column] < 0)
            column_name = column
            df.loc[mask, column_name] = np.nan
        elif must_be_rule is WrongValueNumericRule.MUST_BE_NEGATIVE:  # Implementing rule for must be negative rule
            mask = (df[column] >= 0)
            column_name = column
            df.loc[mask, column_name] = np.nan
        elif must_be_rule is WrongValueNumericRule.MUST_BE_LESS_THAN:  # Implementing rule for must be less than rule
            mask = (df[column] > must_be_rule_optional_parameter)
            column_name = column
            df.loc[mask, column_name] = np.nan
        elif must_be_rule is WrongValueNumericRule.MUST_BE_GREATER_THAN:  # Implementing rule for must be greater than rule
            mask = (df[column] < must_be_rule_optional_parameter)
            column_name = column
            df.loc[mask, column_name] = np.nan

    return df


def fix_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    This method should fix the column in respective to outliers depending on the logic you think best to do.
    Feel free to choose which logic you prefer, but if you are in doubt, use the simplest one to remove the row
    of the dataframe which is an outlier (note that the issue with this approach is when the dataset is small,
    dropping rows will make it even smaller).
    Remember that some datasets are large, and some are small, so think wisely on how to calculate outliers
    and when to remove/replace them. Also remember that some columns are numeric, some are categorical, some are binary
    and some are datetime. Use the methods in b_dataset_profile to your advantage!
    :param df: Dataset
    :param column: the column to be investigated and fixed
    :return: The dataset with fixed column
    """
    # Calculating Outliers using interquartile range
    if column in get_numeric_columns(df):
        Q1 = df[column].quantile(.25)
        Q3 = df[column].quantile(.75)
        IQR = Q3 - Q1
        lowerBound = Q1 - 1.5 * IQR  # Upperbound
        upperBound = Q3 + 1.5 * IQR  # Lowerbound

        mask = (df[column] < lowerBound) | (df[column] > upperBound)
        column_name = column
        df.loc[mask, column_name] = np.nan

    return df


def fix_nans(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    This method should fix all nans (missing data) depending on the logic you think best to do
    Remember that some datasets are large, and some are small, so think wisely on when to use each possible
    removal/fix/replace of nans. Also remember that some columns are numeric, some are categorical, some are binary
    and some are datetime. Use the methods in b_dataset_profile to your advantage!
    :param df: Dataset
    :param column: the column to be investigated and fixed
    :return: The fixed dataset
    """
    # Replacing nan in numrical columns with mean
    if column in get_numeric_columns(df):
        df[column] = df[column].fillna(df[column].mean())
    elif column in get_binary_columns(df):
        df[column] = df[column].fillna(df[column].mode()[0])  # Replacing nan in binary columns with mode
    elif column in get_text_categorical_columns(df):
        df[column] = df[column].fillna(df[column].mode()[0])  # Replacing nan in textual columns with mean

    return df


def normalize_column(df_column: pd.Series) -> pd.Series:
    """
    This method should recalculate all values of a numeric column and normalise it between 0 and 1.
    :param df_column: Dataset's column
    :return: The column normalized
    """
    # Normalizing the columns b/w 0 to 1
    df_column = (df_column - df_column.min()) / (df_column.max() - df_column.min())

    # if pd.api.types.is_numeric_dtype(df_column):
    #     df_column = (df_column - df_column.min()) / (df_column.max() - df_column.min())

    return df_column


def standardize_column(df_column: pd.Series) -> pd.Series:
    """
    This method should recalculate all values of a numeric column and standardize it between -1 and 1 with its average at 0.
    :param df_column: Dataset's column
    :return: The column standardized
    """
    df_column = df_column / df_column.abs().max()
    return df_column


def calculate_numeric_distance(df_column_1: pd.Series, df_column_2: pd.Series, distance_metric: DistanceMetric) -> pd.Series:
    """
    This method should calculate the distance between two numeric columns
    :param df_column_1: Dataset's column
    :param df_column_2: Dataset's column
    :param distance_metric: One of DistanceMetric, and for each one you should implement its logic
    :return: A new 'column' with the distance between the two inputted columns
    """
    # Calculating the Euclidean distance
    if pd.api.types.is_numeric_dtype(df_column_1) and pd.api.types.is_numeric_dtype(df_column_2):
        if distance_metric == DistanceMetric.EUCLIDEAN:
            diff = np.sqrt(np.square((df_column_1 - df_column_2)))
            pdSeries = pd.Series(diff)

            return pdSeries
        if distance_metric == DistanceMetric.MANHATTAN:  # Calculating the Manhattan distance
            diff = (df_column_1 - df_column_2)
            pdSeries = pd.Series(diff).abs()

            return pdSeries


def calculate_binary_distance(df_column_1: pd.Series, df_column_2: pd.Series) -> pd.Series:
    """
    This method should calculate the distance between two binary columns.
    Choose one of the possibilities shown in class for future experimentation.
    :param df_column_1: Dataset's column
    :param df_column_2: Dataset's column
    :return: A new 'column' with the distance between the two inputted columns
    """
    # Calculating Binary distance
    answer = []
    if pd.api.types.is_bool(df_column_1) and pd.api.types.is_bool(df_column_2):

        for i in range(df_column_1.size):
            if df_column_1[i] == df_column_2[i]:
                answer.append(0)
            else:
                answer.append(1)

    pdSeries = pd.Series(answer)

    print(pdSeries)

    return pdSeries


