import csv
import math
import matplotlib.pyplot as plt

def load_csv(filepath):
    with open(filepath, 'r') as file:
        csv_reader = csv.reader(file)
        data = list(csv_reader)
    print(f"Loaded {len(data)-1} rows from {filepath}")
    return data[1:]  # Skipping header

def preprocess_data(data):
    X = [list(map(float, row[:-1])) for row in data]  # Convert features to floats
    y = [int(row[-1]) for row in data]  # Convert labels to integers
    print(f"Preprocessed data: {len(X)} samples")
    return X, y

def dot_product(v1, v2):
    return sum(x*y for x, y in zip(v1, v2))

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

class LogisticRegressionSGD:
    def __init__(self, learning_rate=0.01, iterations=200):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
    
    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        self.weights = [0.0] * n_features
        self.loss_history = []  # Store loss at each iteration
        
        for _ in range(self.iterations):
            current_loss = 0
            for i in range(n_samples):
                linear_combination = dot_product(X[i], self.weights)
                y_predicted = sigmoid(linear_combination)
                update = [self.learning_rate * (y[i] - y_predicted) * x_i for x_i in X[i]]
                self.weights = [w + u for w, u in zip(self.weights, update)]
                # Calculate the loss for the current sample
                current_loss += y[i] * math.log(y_predicted) + (1 - y[i]) * math.log(1 - y_predicted)
            
            # Calculate the average loss over all samples
            current_loss = -current_loss / n_samples
            self.loss_history.append(current_loss)
            
            if (_ + 1) % 10 == 0:
                print(f"Iteration {_ + 1}/{self.iterations}, Loss: {current_loss}")
    
    def predict_prob(self, X):
        return [sigmoid(dot_product(x, self.weights)) for x in X]
    
    def predict(self, X):
        probabilities = self.predict_prob(X)
        return [1 if prob >= 0.5 else 0 for prob in probabilities]

def evaluate_metrics(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    
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
    return accuracy, precision_pos, recall_pos, f1_score_pos, precision_neg, recall_neg, f1_score_neg

# Load and preprocess the datasets
train_data = load_csv('./train-1.csv')
test_data = load_csv('./test-1.csv')

X_train, y_train = preprocess_data(train_data)
X_test, y_test = preprocess_data(test_data)

# Initialize and train the Logistic Regression model
model = LogisticRegressionSGD()
print("Starting training...")
model.fit(X_train, y_train)

# Evaluate on training set
print("\nEvaluating on training set...")
train_predictions = model.predict(X_train)
evaluate_metrics(y_train, train_predictions)

# Predict on test set and evaluate
print("\nPredicting on test set...")
test_predictions = model.predict(X_test)
evaluate_metrics(y_test, test_predictions)

total_cost = model.loss_history[-1]
print(f"Total cost of the model: {total_cost}")

# Optionally, you can visualize the loss over iterations using matplotlib
plt.plot(model.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Log Loss')
plt.title('Training Log Loss Over Iterations')
plt.show()

'''
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