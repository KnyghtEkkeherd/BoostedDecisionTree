from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import pandas

class BoostedDecisionTreeClassifier:
    def __init__(self, n_estimators: int=50, max_depth: int=2, random_state: int=None, n_components: int=None):
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

    def fit(self, X: pandas.DataFrame, y: pandas.DataFrame) -> None:
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

    def predict(self, X: pandas.DataFrame) -> list:
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
    
    def preprocess_and_predict(self, new_data: pandas.DataFrame) -> list:
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