import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class BoostedDecisionTreeClassifier:
    def __init__(self, n_estimators=50, max_depth=None, random_state=None, n_components=None, normalize=False):
        """
        Initialize the Boosted Decision Tree Classifier.

        Args:
        - n_estimators: Number of weak classifiers (Decision Trees) to boost.
        - max_depth: Maximum depth of the base Decision Trees.
        - random_state: Random seed for reproducibility.
        - n_components: Number of PCA components (if None, PCA is not used).
        - normalize: Whether to normalize the data before PCA.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_components = n_components
        self.normalize = normalize
        self.ada_clf = None
        self.pca = None
        self.scaler = None

    def fit(self, X, y):
        """
        Fit the classifier to the training data and optionally precompute PCA with normalization.

        Args:
        - X: Training data features (Numpy array or Pandas DataFrame).
        - y: Training data labels (Numpy array or Pandas Series).
        """
        if self.n_components is not None:
            if self.normalize:
                self.scaler = StandardScaler()
                X = self.scaler.fit_transform(X)

            self.pca = PCA(n_components=self.n_components)
            X = self.pca.fit_transform(X)

        base_estimator = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
        self.ada_clf = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        self.ada_clf.fit(X, y)

    def predict(self, X):
        """
        Make predictions on new data after PCA transformation.

        Args:
        - X: New data points to classify (Numpy array or Pandas DataFrame).

        Returns:
        - predicted_labels: List of predicted class labels for the new data.
        """
        if self.ada_clf is None:
            raise ValueError("Classifier has not been trained. Call the 'fit' method first.")

        if self.n_components is not None:
            if self.normalize:
                X = self.scaler.transform(X)
            X = self.pca.transform(X)

        predictions = self.ada_clf.predict(X)
        predicted_labels = predictions.tolist()
        return predicted_labels
