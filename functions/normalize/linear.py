import pandas as pd
import numpy as np
import logging, sys

# Columns of factors must be a subset of columns of df with the exception of 
# ignore list.


def norm(
        raw_df, 
        check=False, 
        ignore: list = [],
        factors=None,
        rename: dict = {},
    ):

    df = raw_df.copy()
    for column in df.columns:
        if column in rename:
            df.rename(columns={column: rename[column]}, inplace=True)

    if isinstance(factors, pd.DataFrame):     # TODO: may need to check if factors is a dataframe or pandas.core.frame.DataFrame
        for column in df.columns:
            if column in ignore:
                logging.debug(f'column {column} is in ignore list, skipping normalization')
                continue

            if column in factors.columns:            
                logging.debug(f'column {column} is in factors, applying normalization')
                df[column] = (df[column] + factors.loc['shift', column]) * factors.loc['scale', column]
            else:
                logging.debug(f'column {column} is not in factors, skipping normalization')

        rename = {v: k for k, v in rename.items()}
        for column in df.columns:
            if column in rename:
                df.rename(columns={column: rename[column]}, inplace=True)

        return df, factors
    

    # two modes:
    # 1. scale entire column ()

    # factors are a df that describes the column-wise normalization
    factors = pd.DataFrame(columns=df.columns, index=['scale', 'shift'])

    for column in df.columns:
        if column in ignore:
            logging.debug(f'column {column} is in ignore list, skipping normalization')
            continue

        # find the min and max of the column
        min = float(df[column].min())
        max = float(df[column].max())

        logging.debug(f'column = {column}')
        logging.debug(f'type(min) = {type(min)}')
        logging.debug(f'min = {min}')
        logging.debug(f'type(max) = {type(max)}')

        if isinstance(min, bool) or isinstance(max, bool):
            logging.debug(f'column {column} is boolean, skipping normalization')
            continue

        if isinstance(min, np.bool_) or isinstance(max, np.bool_):
            logging.debug(f'column {column} is boolean, skipping normalization')
            continue

        # there are two modes:
        # 1. scale the entire column and let shift be 0
        # 2. scale the column and shift it so that the min/max is 0,
        #    depending on whether the 
        # NOTE: Both modes are constrained [-1, 1] and if it is mod 2, then
        #       it is constrained to either [-1, 0] or [0, 1]
        mode = 0

        # mode 1
        if max > 0 and min < 0:
            mode = 1
            if np.abs(max) >= np.abs(min):
                # scale the columns using max
                scale = 1 / np.abs(max)
                shift = 0
                df[column] = df[column] * scale
                factors[column] = [scale, shift]

            elif np.abs(max) < np.abs(min):
                # scale the columns using min
                scale = 1 / np.abs(min)
                shift = 0
                df[column] = df[column] * scale
                factors[column] = [scale, shift]
        # mode 2 (positive)
        elif max > 0 and min >= 0:
            mode = 2
            # both min and max are positive, so shift is negative
            shift = - min
            scale = np.abs(1 / (max - min))
            df[column] = (df[column] + shift) * scale
            factors[column] = [scale, shift]
        # mode 2 (negative)
        elif max <= 0 and min < 0:
            mode = 2
            # both min and max are negative, so shift is positive
            shift = - max # NOTE: max is negative, so this makes shift positive
            scale = np.abs(1 / (min - max))
            df[column] = (df[column] + shift) * scale
            factors[column] = [scale, shift]
        else:
            shift = None
            scale = None
            # raise ValueError("Normalization failed")

        if mode == 0:
            raise ValueError("Normalization failed (mode 0)")

        logging.debug("Normalizing column: {}".format(column))
        logging.debug("Normalization mode: {}".format(mode))
        logging.debug("Shift: {}".format(shift))
        logging.debug("Scale: {}".format(scale))
        logging.debug("New Min: {}".format(df[column].min()))
        logging.debug("New Max: {}".format(df[column].max()))


        roundto = 2

        if check:
            if mode == 1:
                # check that the min and max are -1 and 1
                if round(df[column].min(), roundto) != -1 and round(df[column].max(), roundto) != 1:
                    raise ValueError("Normalization failed (mode 1)")
            elif mode == 2:
                # check that the min and max are either 0 and 1 or -1 and 0
                if [round(df[column].min(), roundto), round(df[column].max(), roundto)] not in [[0, 1], [-1, 0]]:
                    raise ValueError("Normalization failed (mode 2)")
                
    # if 'TV_z' in df.columns:
    #     df['TV_z'] = df['TV_z'] * abs(df['TV_z'])**(-2/5) # for rescaling df['TV_z']**(1/3)
    #     count_nan = df['TV_z'].isna().sum()
    #     if count_nan > 0:
    #         logging.warning(f'(norm) Number of NaN values in TV_z: {count_nan}')

    return df, factors


# TODO: update this to use pkl instead of csv by default
# but maybe allow for csv when argument is passed
def load_factors(factorfile):

    factors = pd.read_csv(factorfile) # this should be the factors dataframe

    assert factors.index == ['scale', 'shift'], "Factors file is not formatted correctly"

    return factors


def denorm(
        df, 
        factors, 
        ignore: list = [],
        rename: dict = {},
    ):
    df_cols = [col for col in df.columns if col not in ignore]
    factors_cols = [col for col in factors.columns if col not in ignore]

    # TODO: add rename functionality to denorm function

    if not set(df_cols).issubset(set(factors_cols)):
        logging.error("Columns in dataframe are not a subset of the columns in the normalization factors dataframe.")
        logging.debug('Could not find every column of the dataframe to be denormalized in the factors dataframe.')
        logging.debug(f"Columns in dataframe: {df.columns}")
        logging.debug(f"Columns in factors dataframe: {factors.columns}")
        logging.debug(f'Ingore: {ignore}')
        logging.error("Exiting...")
        sys.exit(1)

    # if 'TV_z' in df.columns:
    #     df['TV_z'] = df['TV_z']**(-3/5)  # to undo rescaling df['TV_z']**(1/3)
    #     count_nan = df['TV_z'].isna().sum()
    #     if count_nan > 0:
    #         logging.warning(f'(denorm) Number of NaN values in TV_z: {count_nan}')

    for column in df.columns:
        if column in df_cols:
            scale = factors[column]['scale']
            shift = factors[column]['shift']
            # df[column] = (df[column] - shift) / scale
            df[column] = (df[column] / scale) - shift
        if column in ignore:
            df[column] = df[column]
        else:
            logging.debug(f'column {column} is not in the dataframe to be denormalized and is not in the ignore list.')
            if 'TV_z' in factors.columns:
                logging.debug(f'However, TV_z is in the factors dataframe.')

    return df


# TODO: update this to use pkl instead of csv by default
# but maybe allow for csv when argument is passed
def save_factors(filename:str, factors:pd.DataFrame):
    # save the factors to a csv file
    return factors.to_csv(filename)


# TODO: make a normalization test case for normalizing the data, 
#       denormalizing, and checking that the bin counts are identical across
#       original and denormalized data.