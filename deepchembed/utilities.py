import numpy as np
import pandas as pd

def clean_out_of_bound(bio_deg):
    """
    clean the bio degradation part, if negative, treat as 0
    if above 100, treat as 100
    ----
    Args:
    bio_deg: pd.series or list or numpy.ndarray
    ----
    Return:
    cleaned pd.series
    """
    cleaned_bio = []
    for i in bio_deg:
        if i < 0:
            i = 0
        elif i > 100:
            i = 100
        cleaned_bio.append(i)
    return pd.Series(cleaned_bio)


def bi_class(raw, boundary):
    """
    divide raw input into two classes, based on selected boundary
    """
    bi_class = pd.Series([0 if i < boundary else 1 for i in raw])
    return bi_class



def check_dtypes_count(df):
    """
    Quickly check for unique data types in a dataframe and return
    counts for each type.
    ----
    Args:
        pd.dataframe
    ----
    Return
        dtypes and counts: tuple of two arrays
    """
    return np.unique(df.dtypes, return_counts=True)

def dedup_input_cols(df):
    """
    return columns that has distinct input
    """
    return df.loc[:,df.nunique()>1]

def assign_class(num, cuts):
    """
    num: int/float target to be assigned to classes
    cuts: list(of float/int) or np.ndarray to be used as cut-edges between
        classes, no start/end value
    """
    assert len(cuts) >0; "cuts can not be empty"

    for i in range(len(cuts)):
        if num <=cuts[i]:
            return i

    return i+1


def divide_classes(lst, cuts):
    """
    lst: pd.dataframe(int/float)
    cuts: list(of float/int) or np.ndarray to be used as cut-edges between
        classes, no start/end value
    """
    cls = [assign_class(num,cuts) for num in lst]
    return pd.DataFrame(cls)
