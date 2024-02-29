import numpy as np
import pandas as pd

# Load the datasets
train_data_path = './train-1.csv'
test_data_path = './test-1.csv'

train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

# Display the first few rows of each dataset to understand their structure
train_df.head(), test_df.head()

class LogisticRegressionSGD:
    def __init__(self, learning_rate=0.01, iterations=200):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights to zeros
        self.weights = np.zeros(n_features)
        
        # Stochastic Gradient Descent
        for _ in range(self.iterations):
            for i in range(n_samples):
                # Compute the linear combination
                linear_combination = np.dot(X[i], self.weights)
                # Predict using sigmoid
                y_predicted = self.sigmoid(linear_combination)
                # Update weights
                update = self.learning_rate * (y[i] - y_predicted) * X[i]
                self.weights += update
    
    def predict_prob(self, X):
        linear_combination = np.dot(X, self.weights)
        return self.sigmoid(linear_combination)
    
    def predict(self, X):
        probabilities = self.predict_prob(X)
        return [1 if i >= 0.5 else 0 for i in probabilities]

# Preparing the datasets
X_train = train_df.drop('label', axis=1).values
y_train = train_df['label'].values

X_test = test_df.drop('label', axis=1).values
y_test = test_df['label'].values

# Initialize and train the Logistic Regression model
model = LogisticRegressionSGD()
model.fit(X_train, y_train)

# Predict on test set
predictions = model.predict(X_test)

# Evaluation
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

accuracy = accuracy_score(y_test, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='binary')
conf_matrix = confusion_matrix(y_test, predictions)

accuracy, precision, recall, f1, conf_matrix
