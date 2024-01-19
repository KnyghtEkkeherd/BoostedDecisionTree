#!/usr/bin/env python
# coding: utf-8

# **MAIN BODY OF DATA CLEANING**



import numpy as np
from scipy import misc
from importlib import reload
import random
import pprint as pp
import pandas as pd
import pprint as pp
import math
import matplotlib.pyplot as plt



def calculateMean(data):
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

def getDataFrame(file_path):
    # Load the CSV file into a DataFrame, assuming the first row contains column names
    df = pd.read_csv(file_path, index_col=0)  # Assuming the index is in the first column
    return df

def get_values(dataframe, column):
    """
    Get all possible parameters for the given feature
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

def change_type(dataframe, column_name):
    """
    Change the value of the parameters that should be float from string
    """
    dataframe[column_name] = pd.to_numeric(dataframe[column_name], errors='coerce')
    return dataframe

def assignTargetClasses(dataframe):
    """
    Give the string y-values a class of a number 1, 2, 3, ... 
    """
    y_values = get_values(dataframe, 'y')
    dataframe['y'] = dataframe['y'].map(y_values)
    
    return dataframe, y_values

def assignInputClasses(dataframe, column_name):
    """
    Assign a value to a parameter value that is not a float or int (x7)
    """
    x_values = get_values(dataframe, column_name)
    dataframe[column_name] = dataframe[column_name].map(x_values)
    
    return dataframe


def checkColumnUniqness(dataframe, column_name):
    """
    Check the column to see if the entries in it are the same, if they are, delete the column
    """
    is_same = dataframe[column_name].nunique() == 1

    if is_same:
        dataframe = dataframe.drop(column_name, axis=1)

    return dataframe


# Function to filter rows with correct data types
def filter_correct_data_types(df):
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
        mask &= df[column_name].apply(lambda x: isinstance(x, expected_type))

    if not mask.any():
        print("No rows match the expected data types.")

    return df[mask]


def writeToCsv(dataframe, file_name):

    # Write the DataFrame to a CSV file
    dataframe.to_csv(file_name, index=False)
    print(f"""Dataframe written to {file_name}""")
    return None


def extractData(dataframe):
    """
    Extract inputs and targets from the dataset
    """
    x_columns = dataframe.columns
    inputs = dataframe[x_columns].values
    targets = None
    if 'y' in dataframe.columns:
        targets= dataframe['y'].values
    
    return inputs, targets


def clearData(dataframe):
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
    
    # See of entries in a given column are the same, if so, delete the column
    """
    for column_name in dataframe.columns:
        dataframe = checkColumnUniqness(dataframe, column_name)
    """
    return dataframe


def clearDataToClassify(dataframe):
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


df = getDataFrame("TrainOnMe.csv")
cleared_dataframe = clearData(df)
writeToCsv(cleared_dataframe, "ClearedSet.csv")


training_inputs, training_targets = extractData(cleared_dataframe)


# **MAIN BODY OF THE CLASSIFIERS**

# In[387]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


# In[388]:


# Get the training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(training_inputs, training_targets, test_size=0.5)


# **Boosted Decision Tree Classifier | NO CROSS VALIDATION**

# In[419]:


pca = PCA(n_components=11)  # Adjust the number of components as needed
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Create an AdaBoostClassifier with a DecisionTree base estimator
base_estimator = DecisionTreeClassifier(max_depth=4)
ada_clf = AdaBoostClassifier(estimator=base_estimator, n_estimators=50)

# Fit the model to the training data
ada_clf.fit(X_train_pca, y_train)

# Make predictions on the test set
y_pred = ada_clf.predict(X_test_pca)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
accuracy_percent = accuracy * 100
print(f'Accuracy: {accuracy_percent:.2f}%')


# **Cross Validation of the Classifiers**
# **Boosted Decision Tree | Test ratio parameter in data selection**

# In[390]:


ITER_NUM = 20
TEST_RATIO = np.linspace(0.05, 0.95, 19)

# Save the errors after checking each of them through cross validation
testing_ratio_errors = []

# Randomly select the training and testing data and iterate through it
# TODO: implement the iteration and saving the errors for each parameter measured

for n in TEST_RATIO:
    iteration_error = []
    for i in range(ITER_NUM):
        X_train, X_test, y_train, y_test = train_test_split(training_inputs, training_targets, test_size=n)
        pca = PCA(n_components=11)  # Adjust the number of components as needed
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # Create an AdaBoostClassifier with a DecisionTree base estimator
        base_estimator = DecisionTreeClassifier(max_depth=2)
        ada_clf = AdaBoostClassifier(estimator=base_estimator, n_estimators=50)

        # Fit the model to the training data
        ada_clf.fit(X_train_pca, y_train)

        # Make predictions on the test set
        y_pred = ada_clf.predict(X_test_pca)

        # Evaluate the classifier
        accuracy = accuracy_score(y_test, y_pred)
        
        # add the score to the list of scores
        iteration_error.append(1-accuracy)
    testing_ratio_errors.append(iteration_error)


# In[391]:


x = np.linspace(0.05, 0.95, 19)
y, e = calculatePlotValues(testing_ratio_errors)
print(len(x))
print(len(y))

plt.errorbar(x, y, yerr=e, linestyle='solid', marker='.', label='Error')
plt.ylabel('Error fraction')
plt.xlabel("Testing Fraction")
plt.title(f"""Boosted Decision Tree error fraction based the testing data ratio""")
plt.legend()


# **Cross Validation of the Classifiers**
# **Boosted Decision Tree | Test PCA parameter number**

# In[392]:


ITER_NUM = 20
PCA_NUM = range(1, 13)

# Save the errors after checking each of them through cross validation
pca_errors = []

# Randomly select the training and testing data and iterate through it
# TODO: implement the iteration and saving the errors for each parameter measured

for n in PCA_NUM:
    iteration_error = []
    for i in range(ITER_NUM):
        X_train, X_test, y_train, y_test = train_test_split(training_inputs, training_targets, test_size=0.5)
        pca = PCA(n_components=n)  # Adjust the number of components as needed
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # Create an AdaBoostClassifier with a DecisionTree base estimator
        base_estimator = DecisionTreeClassifier(max_depth=2)
        ada_clf = AdaBoostClassifier(estimator=base_estimator, n_estimators=50)

        # Fit the model to the training data
        ada_clf.fit(X_train_pca, y_train)

        # Make predictions on the test set
        y_pred = ada_clf.predict(X_test_pca)

        # Evaluate the classifier
        accuracy = accuracy_score(y_test, y_pred)
        
        # add the score to the list of scores
        iteration_error.append(1-accuracy)
    pca_errors.append(iteration_error)


# In[393]:


x = PCA_NUM
y, e = calculatePlotValues(pca_errors)
print(len(x))
print(len(y))

plt.errorbar(x, y, yerr=e, linestyle='solid', marker='.', label='Error')
plt.ylabel('Error fraction')
plt.xlabel("Testing Fraction")
plt.title(f"""Boosted Decision Tree error fraction based on the pca value""")
plt.legend()


# **Cross Validation of the Classifiers**
# **Boosted Decision Tree | Test PCA parameter number**

# In[394]:


ITER_NUM = 40
DEPTH_NUM = range(1, 40)

# Save the errors after checking each of them through cross validation
depth_errors = []

# Randomly select the training and testing data and iterate through it
# TODO: implement the iteration and saving the errors for each parameter measured

for n in DEPTH_NUM:
    iteration_error = []
    for i in range(ITER_NUM):
        X_train, X_test, y_train, y_test = train_test_split(training_inputs, training_targets, test_size=0.5)
        pca = PCA(n_components=11)  # Adjust the number of components as needed
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # Create an AdaBoostClassifier with a DecisionTree base estimator
        base_estimator = DecisionTreeClassifier(max_depth=n)
        ada_clf = AdaBoostClassifier(estimator=base_estimator, n_estimators=50)

        # Fit the model to the training data
        ada_clf.fit(X_train_pca, y_train)

        # Make predictions on the test set
        y_pred = ada_clf.predict(X_test_pca)

        # Evaluate the classifier
        accuracy = accuracy_score(y_test, y_pred)
        
        # add the score to the list of scores
        iteration_error.append(1-accuracy)
    depth_errors.append(iteration_error)


# In[395]:


x = DEPTH_NUM
y, e = calculatePlotValues(depth_errors)
print(len(x))
print(len(y))

plt.errorbar(x, y, yerr=e, linestyle='solid', marker='.', label='Error')
plt.ylabel('Error fraction')
plt.xlabel("Testing Fraction")
plt.title(f"""Boosted Decision Tree error fraction based on the max depth value""")
plt.legend()


# **Boosted Naive Bayes Classifier**

# In[396]:


# Get the training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(training_inputs, training_targets, test_size=0.65)
# Create a PCA instance
pca = PCA(n_components=13)  # Adjust the number of components as needed

# Apply PCA to the training and testing data
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Create a Gaussian Naive Bayes classifier
naive_bayes_clf = GaussianNB()

# Create an AdaBoostClassifier with Gaussian Naive Bayes as the base estimator
ada_nb_clf = AdaBoostClassifier(estimator=naive_bayes_clf, n_estimators=1)

# Fit the model to the training data
ada_nb_clf.fit(X_train_pca, y_train)

# Make predictions on the test set
y_pred = ada_nb_clf.predict(X_test_pca)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
accuracy = accuracy*100
print(f'Accuracy: {accuracy:.2f}%')


# **Cross Validation of the Classifiers Boosted Naive Bayes | Test ratio parameter in data selection**

# In[397]:


ITER_NUM = 20
TEST_RATIO = np.linspace(0.05, 0.95, 19)

# Save the errors after checking each of them through cross validation
testing_ratio_errors = []

# Randomly select the training and testing data and iterate through it
# TODO: implement the iteration and saving the errors for each parameter measured

for n in TEST_RATIO:
    iteration_error = []
    for i in range(ITER_NUM):
        X_train, X_test, y_train, y_test = train_test_split(training_inputs, training_targets, test_size=n)
        pca = PCA(n_components=11)  # Adjust the number of components as needed

        # Apply PCA to the training and testing data
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # Create a Gaussian Naive Bayes classifier
        naive_bayes_clf = GaussianNB()

        # Create an AdaBoostClassifier with Gaussian Naive Bayes as the base estimator
        ada_nb_clf = AdaBoostClassifier(estimator=naive_bayes_clf, n_estimators=1)

        # Fit the model to the training data
        ada_nb_clf.fit(X_train_pca, y_train)

        # Make predictions on the test set
        y_pred = ada_nb_clf.predict(X_test_pca)

        # Evaluate the classifier
        accuracy = accuracy_score(y_test, y_pred)
        
        # add the score to the list of scores
        iteration_error.append(1-accuracy)
    testing_ratio_errors.append(iteration_error)


# In[398]:


x = np.linspace(0.05, 0.95, 19)
y, e = calculatePlotValues(testing_ratio_errors)
print(len(x))
print(len(y))

plt.errorbar(x, y, yerr=e, linestyle='solid', marker='.', label='Error')
plt.ylabel('Error fraction')
plt.xlabel("Testing Fraction")
plt.title(f"""Boosted Naive Bayes error fraction based the testing data ratio""")
plt.legend()


# **Cross Validation Boosted Naive Bayes | Number of estimators**

# In[399]:


ITER_NUM = 40
ESTIMATOR_NUM = range(1, 100)

# Save the errors after checking each of them through cross validation
estimator_errors = []

# Randomly select the training and testing data and iterate through it
# TODO: implement the iteration and saving the errors for each parameter measured

for n in ESTIMATOR_NUM:
    iteration_error = []
    for i in range(ITER_NUM):
        X_train, X_test, y_train, y_test = train_test_split(training_inputs, training_targets, test_size=0.65)
        pca = PCA(n_components=11)  # Adjust the number of components as needed

        # Apply PCA to the training and testing data
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # Create a Gaussian Naive Bayes classifier
        naive_bayes_clf = GaussianNB()

        # Create an AdaBoostClassifier with Gaussian Naive Bayes as the base estimator
        ada_nb_clf = AdaBoostClassifier(estimator=naive_bayes_clf, n_estimators=n)

        # Fit the model to the training data
        ada_nb_clf.fit(X_train_pca, y_train)

        # Make predictions on the test set
        y_pred = ada_nb_clf.predict(X_test_pca)

        # Evaluate the classifier
        accuracy = accuracy_score(y_test, y_pred)
        
        # add the score to the list of scores
        iteration_error.append(1-accuracy)
    estimator_errors.append(iteration_error)


# In[ ]:


x = ESTIMATOR_NUM
y, e = calculatePlotValues(estimator_errors)
print(len(x))
print(len(y))

plt.errorbar(x, y, yerr=e, linestyle='solid', marker='.', label='Error')
plt.ylabel('Error fraction')
plt.xlabel("Testing Fraction")
plt.title(f"""Boosted Naive Bayes error fraction based the number of estimators""")
plt.legend()


# **Cross Validation Boosted Naive Bayes | PCA**

# In[ ]:


ITER_NUM = 40
PCA_NUM = range(1, 14)

# Save the errors after checking each of them through cross validation
pca_errors = []

# Randomly select the training and testing data and iterate through it
# TODO: implement the iteration and saving the errors for each parameter measured

for n in PCA_NUM:
    iteration_error = []
    for i in range(ITER_NUM):
        X_train, X_test, y_train, y_test = train_test_split(training_inputs, training_targets, test_size=0.65)
        pca = PCA(n_components=n)  # Adjust the number of components as needed

        # Apply PCA to the training and testing data
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # Create a Gaussian Naive Bayes classifier
        naive_bayes_clf = GaussianNB()

        # Create an AdaBoostClassifier with Gaussian Naive Bayes as the base estimator
        ada_nb_clf = AdaBoostClassifier(estimator=naive_bayes_clf, n_estimators=1)

        # Fit the model to the training data
        ada_nb_clf.fit(X_train_pca, y_train)

        # Make predictions on the test set
        y_pred = ada_nb_clf.predict(X_test_pca)

        # Evaluate the classifier
        accuracy = accuracy_score(y_test, y_pred)
        
        # add the score to the list of scores
        iteration_error.append(1-accuracy)
    pca_errors.append(iteration_error)


# In[ ]:


x = PCA_NUM
y, e = calculatePlotValues(pca_errors)
print(len(x))
print(len(y))

plt.errorbar(x, y, yerr=e, linestyle='solid', marker='.', label='Error')
plt.ylabel('Error fraction')
plt.xlabel("Testing Fraction")
plt.title(f"""Boosted Naive Bayes error fraction based on the pca value""")
plt.legend()


# **Classify the EvaluateOnMe.csv** |
# **Boosted Decision Tree Classifier**

# In[ ]:


import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

class BoostedDecisionTreeClassifier:
    def __init__(self, n_estimators=50, max_depth=2, random_state=None, n_components=None):
        """
        Initialize the Boosted Decision Tree Classifier.

        Args:
        - n_estimators: Number of weak classifiers (trees) to boost
        - max_depth: Maximum depth of the decision tree base estimator
        - random_state: Random seed for reproducibility
        - n_components: Number of PCA components (if None, PCA is not used)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_components = n_components
        self.ada_clf = None
        self.pca = None

    def fit(self, X, y):
        """
        Fit the classifier to the training data.

        Args:
        - X: Training data features (Numpy array or Pandas DataFrame)
        - y: Training data labels (Numpy array or Pandas Series)
        """
        if self.n_components is not None:
            self.pca = PCA(n_components=self.n_components)
            X = self.pca.fit_transform(X)
            #print(np.shape(X))

        base_estimator = DecisionTreeClassifier(max_depth=self.max_depth)
        self.ada_clf = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        self.ada_clf.fit(X, y)

    def predict(self, X):
        """
        Make predictions on new data.

        Args:
        - X: New data points to classify (Numpy array or Pandas DataFrame)

        Returns:
        - predicted_labels: List of predicted class labels for the new data
        """
        if self.ada_clf is None:
            raise ValueError("Classifier has not been trained. Call the 'fit' method first.")
        
        if self.n_components is not None:
            X = self.pca.transform(X)

        predictions = self.ada_clf.predict(X)
        predicted_labels = predictions.tolist()
        return predicted_labels
    
    def preprocess_and_predict(self, new_data):
        """
        Preprocess new data and make predictions.

        Args:
        - new_data: New data points to classify (Numpy array or Pandas DataFrame)

        Returns:
        - predicted_labels: List of predicted class labels for the new data
        """
        if self.n_components is not None:
            self.pca = PCA(n_components=self.n_components)
            new_data = self.pca.fit_transform(new_data)

        predictions = self.ada_clf.predict(new_data)
        predicted_labels = predictions.tolist()
        return predicted_labels


# In[420]:


# predict the labels using the optimal boosted decision tree classifier
data_to_classify = getDataFrame("EvaluateOnMe.csv")


# In[421]:


data_to_classify = clearDataToClassify(data_to_classify)
inputs_to_classify = extractData(data_to_classify)[0]


# In[422]:


decision_tree_classifier = BoostedDecisionTreeClassifier(n_estimators=50, max_depth=4, n_components=11)
decision_tree_classifier.fit(training_inputs, training_targets)


# In[423]:


predictions = decision_tree_classifier.preprocess_and_predict(data_to_classify)


# In[443]:


def reverse_dict(original_dict):
    reversed_dict = {v: k for k, v in original_dict.items()}
    return reversed_dict


# In[449]:


def write_labels_to_file(labels, file_path):
    with open(file_path, 'w') as file:
        for label in labels:
            file.write(str(label) + '\n')


# In[450]:


def reverseTargetAssignment(predictedLabels=None):
    
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
    
write_labels_to_file(reverseTargetAssignment(predictions), "predictedLabels.txt")


# In[ ]:




