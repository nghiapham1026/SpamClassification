import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_csv(filepath):
    df = pd.read_csv(filepath)
    print(f"Loaded {df.shape[0]} rows from {filepath}")
    # Splits the DataFrame into features (X) and labels (y), returning them as NumPy arrays.
    return df.drop(df.columns[-1], axis=1).values, df[df.columns[-1]].values

# Preprocesses the data to prepare it for training/testing.
def preprocess_data(X, y):
    # Pandas and NumPy automatically handle type conversion
    positive_count = np.sum(y == 1)  # Count of positive instances.
    negative_count = np.sum(y == 0)  # Count of negative instances.
    print(f"Preprocessed data: {X.shape[0]} samples")
    print(f"Positive instances: {positive_count}, Negative instances: {negative_count}")
    return X, y.astype(int)  # Ensure labels are integers for compatibility.

# Defines a logistic regression model using stochastic gradient descent (SGD) for optimization.
class LogisticRegressionSGD:
    def __init__(self, learning_rate=0.01, iterations=200):
        self.learning_rate = learning_rate  # Learning rate for gradient descent.
        self.iterations = iterations  # Number of iterations to run gradient descent.
        self.weights = None  # Placeholder for model's weights.

    # Trains the logistic regression model on the dataset X with labels y.
    def fit(self, X, y):
        n_samples, n_features = X.shape  # Determines sample size and feature count.
        self.weights = np.zeros(n_features)  # Initializes weights to zeros.
        self.loss_history = []  # To record loss at each iteration.

        # Main training loop.
        for _ in range(self.iterations):
            current_loss = 0  # To accumulate loss over this iteration.
            for i in range(n_samples):
                linear_combination = np.dot(X[i], self.weights)
                y_predicted = 1 / (1 + np.exp(-linear_combination))
                update = self.learning_rate * (y[i] - y_predicted) * X[i]
                self.weights += update  # Update weights.
                # Loss calculation using the log loss formula.
                current_loss += y[i] * np.log(y_predicted) + (1 - y[i]) * np.log(1 - y_predicted)
            current_loss = -current_loss / n_samples  # Average loss over all samples.
            self.loss_history.append(current_loss)  # Record for plotting.
            if (_ + 1) % 10 == 0:  # Print loss every 10 iterations.
                print(f"Iteration {_ + 1}/{self.iterations}, Loss: {current_loss}")

    # Predicts the probability of the positive class for each sample in X.
    def predict_prob(self, X):
        return 1 / (1 + np.exp(-np.dot(X, self.weights)))

    # Uses predict_prob to classify each sample in X as 0 or 1 based on a threshold of 0.5.
    def predict(self, X):
        return (self.predict_prob(X) >= 0.5).astype(int)

# Evaluates the model's performance using accuracy, precision, recall, and F1 score.
def evaluate_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))  # True positives.
    tn = np.sum((y_true == 0) & (y_pred == 0))  # True negatives.
    fp = np.sum((y_true == 0) & (y_pred == 1))  # False positives.
    fn = np.sum((y_true == 1) & (y_pred == 0))  # False negatives.
    # Calculate metrics.
    accuracy = (tp + tn) / len(y_true)
    precision_pos = tp / (tp + fp) if (tp + fp) else 0
    recall_pos = tp / (tp + fn) if (tp + fn) else 0
    f1_score_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos) if (precision_pos + recall_pos) else 0
    precision_neg = tn / (tn + fn) if (tn + fn) else 0
    recall_neg = tn / (tn + fp) if (tn + fp) else 0
    f1_score_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg) if (precision_neg + recall_neg) else 0
    # Print performance metrics.
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Positive (Spam) Precision: {precision_pos:.4f}, Recall: {recall_pos:.4f}, F1 Score: {f1_score_pos:.4f}")
    print(f"Negative (Ham) Precision: {precision_neg:.4f}, Recall: {recall_neg:.4f}, F1 Score: {f1_score_neg:.4f}")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

# Main script: load, preprocess, train, evaluate, and plot training progress.
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
plt.plot(model.loss_history)  # Plot log liss
plt.xlabel('Iteration')
plt.ylabel('Log Loss')
plt.title('Training Log Loss Over Iterations')
plt.show()

'''
Loaded 4459 rows from ./train-1.csv
Loaded 1115 rows from ./test-1.csv
Preprocessed data: 4459 samples
Positive instances: 586, Negative instances: 3873
Preprocessed data: 1115 samples
Positive instances: 161, Negative instances: 954
Starting training...
Iteration 10/200, Loss: 0.1750429972941354
Iteration 20/200, Loss: 0.13750056938544888
Iteration 30/200, Loss: 0.11861972786595598
Iteration 40/200, Loss: 0.10625013527390886
Iteration 50/200, Loss: 0.09717340275983574
Iteration 60/200, Loss: 0.09009117550282754
Iteration 70/200, Loss: 0.08434660412992144
Iteration 80/200, Loss: 0.07955900052779097
Iteration 90/200, Loss: 0.07548798031526859
Iteration 100/200, Loss: 0.07197228837238936
Iteration 110/200, Loss: 0.06889821038663452
Iteration 120/200, Loss: 0.06618213965002254
Iteration 130/200, Loss: 0.06376077173897395
Iteration 140/200, Loss: 0.061585258746558555
Iteration 150/200, Loss: 0.05961734324323421
Iteration 160/200, Loss: 0.05782662206892226
Iteration 170/200, Loss: 0.05618857716482593
Iteration 180/200, Loss: 0.054683160309586414
Iteration 190/200, Loss: 0.05329377554796851
Iteration 200/200, Loss: 0.0520065413616742

Evaluating on training set...
Accuracy: 0.9818
Positive (Spam) Precision: 0.8903, Recall: 0.9829, F1 Score: 0.9343
Negative (Ham) Precision: 0.9974, Recall: 0.9817, F1 Score: 0.9895
Confusion Matrix: TP=576, FP=71, TN=3802, FN=10

Predicting on test set...
Accuracy: 0.9561
Positive (Spam) Precision: 0.8111, Recall: 0.9068, F1 Score: 0.8563
Negative (Ham) Precision: 0.9840, Recall: 0.9644, F1 Score: 0.9741
Confusion Matrix: TP=146, FP=34, TN=920, FN=15
Total cost of the model: 0.0520065413616742
'''