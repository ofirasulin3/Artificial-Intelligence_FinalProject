import numpy as np
import pandas as pd
# from typing import *

def get_data_from_csv(file_name):
    """Returns the data that is saved as a csv file in current folder.
    """
    # Using the function to load the data of example.csv into a Dataframe df
    df = pd.read_csv(file_name)
    # print("Indexes numbers: ")
    # print(df.index)
    # print()
    # print("Columns names: ")
    # print(df.columns)
    # print("Columns dtypes: ")
    # print(df.dtypes)

    # Print the Dataframe
    # print(dataframe)
    # df_ = df.astype(dtype={'diagnosis': np.float64})
    # data_array = df_.to_numpy()
    # data_array = df.to_numpy(dtype={'diagnosis': np.float64})
    data_array = df.to_numpy()

    # data_array = np.loadtxt(file_name, delimiter=',', skiprows=1)
    # print("data_array: ")
    # print(data_array)

    return data_array

    # Return the Dataframe
    # return df
