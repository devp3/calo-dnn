import pandas as pd
import logging

def reorder(df:pd.DataFrame, ordered_columns:list):
    """A function to reorder the columns in df to match the order in the list 
    of columns specified by ordered_columns. 

    Args:
        df (pd.DataFrame): dataframe to reorder
        ordered_columns (list): list of strings specifying the desired column order for df

    Raises:
        ValueError: if the set of columns in df does not match the set of columns in ordered_columns

    Returns:
        pd.DataFrame: dataframe with columns reordered
    """
    # check that the list of columns is the same as the list of columns in the dataframe
    if set(df.keys()) != set(ordered_columns):
        raise ValueError("Columns in dataframe do not match columns in list")
    
    # reorder the columns

    logging.debug("Reordering columns")

    df = df[ordered_columns]

    # NOTE: The purpose of having a separate function for reording the pandas
    # dataframe is really just to make sure the columns in the reordering list 
    # are all in the original dataframe. If they are not, then the function 
    # will raise an error. This is a sanity check. 

    return df    