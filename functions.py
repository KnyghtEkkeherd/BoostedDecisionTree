import numpy as np
import pandas
import math
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

def calculateMean(data: np.array) -> float:
    """
    Calculate the mean of the array

    Args:
        data (np.array): input data array

    Returns:
        float: output mean
    """
    
    sumData = 0
    for i in range(len(data)):
        sumData += data[i]
    mean = sumData / len(data)
    return mean


def calculatePlotValues(error_array):
    y = [] # mean values
    e = [] # standard deviations
    
    for parameter in error_array:
        y.append(calculateMean(parameter))
        e.append(np.std(parameter))
    return y, e


def getDataFrame(file_path: str) -> pandas.DataFrame:
    """
    Get data frame from the file path

    Args:
        file_path (str): path  to the file

    Returns:
        pandas.DataFrame: output data frame
    """
    
    # Load the CSV file into a DataFrame, assuming the first row contains column names
    df = pandas.read_csv(file_path, index_col=0)  # Assuming the index is in the first column
    return df


def get_values(dataframe: pandas.DataFrame, column: str) -> list:
    """
    Get all possible parameters for the given feature

    Args:
        dataframe (pandas.DataFrame): input data frame
        column (str): column name of the input data frame 

    Returns:
        list: unique contents in the column
    """
    
    unique_features = []
    for entry in dataframe.loc[:,column]:
        try:
            if math.isnan(entry):
                continue
        except:
            if (entry not in unique_features):
                unique_features.append(entry)
    unique_features = enumerate(unique_features, 1)
    unique_features = [t[::-1] for t in unique_features]
    unique_features = dict(unique_features)
    return unique_features


def change_type(dataframe: pandas.DataFrame, column_name: str) -> pandas.DataFrame:
    """
    Change the value of the parameters that should be float from string

    Args:
        dataframe (pandas.DataFrame): input data frame
        column_name (str): name of the column

    Returns:
        pandas.DataFrame: data frame with the changed column entry types
    """
    
    dataframe[column_name] = pandas.to_numeric(dataframe[column_name], errors='coerce')
    return dataframe


def assignTargetClasses(dataframe: pandas.DataFrame) -> pandas.DataFrame:
    """
    Give the string y-values a class of a number: 1, 2, 3, ... 

    Args:
        dataframe (pandas.DataFrame): input data frame

    Returns:
        pandas.DataFrame: data frame with assigned target classes
    """
    
    y_values = get_values(dataframe, 'y')
    dataframe['y'] = dataframe['y'].map(y_values)
    
    return dataframe, y_values


def assignInputClasses(dataframe: pandas.DataFrame, column_name: str) -> pandas.DataFrame:
    """
    Assign a value to a parameter value that is not a float or int

    Args:
        dataframe (pandas.DataFrame): input data frame
        column_name (str): name of the column

    Returns:
        pandas.DataFrame: data frame with assigned input classes
    """
    
    x_values = get_values(dataframe, column_name)
    dataframe[column_name] = dataframe[column_name].map(x_values)
    
    return dataframe


def checkColumnUniqness(dataframe: pandas.DataFrame, column_name: str) -> pandas.DataFrame:
    """
    Check the column to see if the entries in it are the same, if they are, delete the column

    Args:
        dataframe (pandas.DataFrame): input data frame
        column_name (str): column name

    Returns:
        pandas.DataFrame: output data frame with reduced number of columns to consider
    """
    
    is_same = dataframe[column_name].nunique() == 1

    if is_same:
        dataframe = dataframe.drop(column_name, axis=1)

    return dataframe


def filter_correct_data_types(dataframe: pandas.DataFrame) -> pandas.DataFrame:
    """
    

    Args:
        dataframe (pandas.DataFrame): _description_

    Returns:
        pandas.DataFrame: data frame with filtered data using a type mask
    """
    
    mask = True
    # Define the expected data types for each column
    expected_data_types = {
        'y': str,
        'x1': float,
        'x2': float,
        'x3': float,
        'x4': float,
        'x5': float,
        'x6': float,
        'x7': str,
        'x8': float,
        'x9': float,
        'x10': float,
        'x11': float,
        'x12': bool,
        'x13': float
    }
    
    for column_name, expected_type in expected_data_types.items():
        mask &= dataframe[column_name].apply(lambda x: isinstance(x, expected_type))

    if not mask.any():
        print("No rows match the expected data types.")

    return dataframe[mask]


def writeToCsv(dataframe: pandas.DataFrame, file_name: str) -> None:
    dataframe.to_csv(file_name, index=False)
    print(f"""Dataframe written to {file_name}""")
    return None


def extractData(dataframe: pandas.DataFrame) -> pandas.DataFrame:
    """
    Extracts inputs and targets from the data set

    Args:
        dataframe (pandas.DataFrame): input data frame

    Returns:
        pandas.DataFrame: inputs and targets data frames
    """
    
    x_columns = dataframe.columns
    inputs = dataframe[x_columns].values
    targets = None
    if 'y' in dataframe.columns:
        targets= dataframe['y'].values
    
    return inputs, targets


def clearData(dataframe: pandas.DataFrame) -> pandas.DataFrame:
    """
    """
    dataframe = dataframe.dropna()
    dataframe = change_type(dataframe, 'x1')
    dataframe = change_type(dataframe, 'x2')
    dataframe = change_type(dataframe, 'x3')
    dataframe = change_type(dataframe, 'x4')
    dataframe = change_type(dataframe, 'x5')
    dataframe = change_type(dataframe, 'x6')
    dataframe = change_type(dataframe, 'x8')
    dataframe = change_type(dataframe, 'x9')
    dataframe = change_type(dataframe, 'x10')
    dataframe = change_type(dataframe, 'x11')
    dataframe['x12'] = dataframe['x12'].astype(bool)
    dataframe = change_type(dataframe, 'x13')
    
    dataframe = filter_correct_data_types(dataframe)
    dataframe, y_values = assignTargetClasses(dataframe)
    # Data specifics that only x7 is a string
    dataframe = assignInputClasses(dataframe, 'x7')
    
    return dataframe


def clearDataToClassify(dataframe: pandas.DataFrame) -> pandas.DataFrame:
    dataframe = change_type(dataframe, 'x1')
    dataframe = change_type(dataframe, 'x2')
    dataframe = change_type(dataframe, 'x3')
    dataframe = change_type(dataframe, 'x4')
    dataframe = change_type(dataframe, 'x5')
    dataframe = change_type(dataframe, 'x6')
    dataframe = change_type(dataframe, 'x8')
    dataframe = change_type(dataframe, 'x9')
    dataframe = change_type(dataframe, 'x10')
    dataframe = change_type(dataframe, 'x11')
    dataframe['x12'] = dataframe['x12'].astype(bool)
    dataframe = change_type(dataframe, 'x13')
    
    # Data specifics that only x7 is a string
    dataframe = assignInputClasses(dataframe, 'x7')
    
    return dataframe


def reverse_dict(original_dict: dict) -> dict:
    """
    Generate reverse dictionary for assigning targets to inputs

    Args:
        original_dict (dict): input dictionary with input-target pairs

    Returns:
        dict: dictionary with target-input pairs
    """
    
    reversed_dict = {v: k for k, v in original_dict.items()}
    return reversed_dict


def write_labels_to_file(labels: list, file_path: str) -> None:
    with open(file_path, 'w') as file:
        for label in labels:
            file.write(str(label) + '\n')


def reverseTargetAssignment(predictedLabels: list=None) -> dict:
    """
    Assign labels to inputs

    Args:
        predictedLabels (list, optional): list of predicted labels. Defaults to None.

    Returns:
        dict: assigned targets to labels
    """
    
    df = getDataFrame("TrainOnMe.csv")
    df = df.dropna()
    df = change_type(df, 'x1')
    df = change_type(df, 'x2')
    df = change_type(df, 'x3')
    df = change_type(df, 'x4')
    df = change_type(df, 'x5')
    df = change_type(df, 'x6')
    df = change_type(df, 'x8')
    df = change_type(df, 'x9')
    df = change_type(df, 'x10')
    df = change_type(df, 'x11')
    df['x12'] = df['x12'].astype(bool)
    df = change_type(df, 'x13')
    
    df = filter_correct_data_types(df)
    y_values = assignTargetClasses(df)[1]
    y_values = reverse_dict(y_values)
    
    output_labels = []
    for label in predictedLabels:
        output_labels.append(y_values[label])
    return output_labels
