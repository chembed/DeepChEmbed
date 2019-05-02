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

    
def cluster_acc(y_true, y_predict):
    """
    Calculate clustering accuracy. 
    ----
    Arguments
        y_true: true labels, numpy.array with shape (n_samples,)
        y_predict: predicted labels, numpy.array with shape (n_samples,)
    ----
    Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_predict.shape[0] == y_true.shape[0]
    compare = y_true == y_predict
    accuarte, count = np.unique(comp, return_counts=True)
    total_evaluation = dict(zip(accuarte, count))
    return total_evaluation[True]/sum(total.values())    
