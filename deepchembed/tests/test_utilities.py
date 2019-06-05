import deepchembed.tests.__init__
import utilities
import pandas as pd

#def test_utilities():

#    return

def test_clean_out_of_bound():
    df = [-9,0,1,99,100,101]
    assert utilities.clean_out_of_bound(df).equals(pd.Series([0,0,1,99,100,100]))
    return

def test_bi_class():

    return

def test_check_dtypes_count():

    return

def test_dedup_input_cols():

    return

def test_assign_class():
    return

def test_divide_classes():

    return



