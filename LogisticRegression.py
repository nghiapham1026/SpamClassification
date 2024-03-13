import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to load data from a CSV file using NumPy for efficiency.
def load_csv(filepath):
    df = pd.read_csv(filepath)
    print(f"Loaded {df.shape[0]} rows from {filepath}")
    return df.drop(df.columns[-1], axis=1).values, df[df.columns[-1]].values

# Function to preprocess data using NumPy for numerical operations.
def preprocess_data(X, y):
    # Convert feature values to floats and label values to integers already done by Pandas and NumPy.
    # Count and print the distribution of classes in the dataset.
    positive_count = np.sum(y == 1)
    negative_count = np.sum(y == 0)
    print(f"Preprocessed data: {X.shape[0]} samples")
    print(f"Positive instances: {positive_count}, Negative instances: {negative_count}")
    return X, y.astype(int)

# Sigmoid activation function optimized with NumPy.
def sigmoid_np(z):
    return 1 / (1 + np.exp(-z))

# Logistic Regression model using Stochastic Gradient Descent with NumPy optimizations.
class LogisticRegressionSGD:
    def __init__(self, learning_rate=0.01, iterations=200):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.loss_history = []
        
        for _ in range(self.iterations):
            current_loss = 0
            for i in range(n_samples):
                linear_combination = np.dot(X[i], self.weights)
                y_predicted = 1 / (1 + np.exp(-linear_combination))
                update = self.learning_rate * (y[i] - y_predicted) * X[i]
                self.weights += update
                current_loss += y[i] * np.log(y_predicted) + (1 - y[i]) * np.log(1 - y_predicted)
            current_loss = -current_loss / n_samples
            self.loss_history.append(current_loss)
            if (_ + 1) % 10 == 0:
                print(f"Iteration {_ + 1}/{self.iterations}, Loss: {current_loss}")
    
    def predict_prob(self, X):
        return 1 / (1 + np.exp(-np.dot(X, self.weights)))
    
    def predict(self, X):
        return (self.predict_prob(X) >= 0.5).astype(int)
    
def evaluate_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    accuracy = (tp + tn) / len(y_true)
    precision_pos = tp / (tp + fp) if (tp + fp) else 0
    recall_pos = tp / (tp + fn) if (tp + fn) else 0
    f1_score_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos) if (precision_pos + recall_pos) else 0
    precision_neg = tn / (tn + fn) if (tn + fn) else 0
    recall_neg = tn / (tn + fp) if (tn + fp) else 0
    f1_score_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg) if (precision_neg + recall_neg) else 0
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Positive (Spam) Precision: {precision_pos:.4f}, Recall: {recall_pos:.4f}, F1 Score: {f1_score_pos:.4f}")
    print(f"Negative (Ham) Precision: {precision_neg:.4f}, Recall: {recall_neg:.4f}, F1 Score: {f1_score_neg:.4f}")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

# Main script starts here with adjustments for NumPy loading and preprocessing.
# Load the training and test datasets using NumPy.
train_features, train_labels = load_csv('./train-1.csv')
test_features, test_labels = load_csv('./test-1.csv')

X_train, y_train = preprocess_data(train_features, train_labels)
X_test, y_test = preprocess_data(test_features, test_labels)

model = LogisticRegressionSGD()
print("Starting training...")
model.fit(X_train, y_train)

print("\nEvaluating on training set...")
train_predictions = model.predict(X_train)
evaluate_metrics(y_train, train_predictions)

print("\nPredicting on test set...")
test_predictions = model.predict(X_test)
evaluate_metrics(y_test, test_predictions)

total_cost = model.loss_history[-1]
print(f"Total cost of the model: {total_cost}")
plt.plot(model.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Log Loss')
plt.title('Training Log Loss Over Iterations')
plt.show()