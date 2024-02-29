import csv
import math

def load_csv(filepath):
    with open(filepath, 'r') as file:
        csv_reader = csv.reader(file)
        data = list(csv_reader)
    print(f"Loaded {len(data)-1} rows from {filepath}")
    return data[1:]  # return data excluding header

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
        n_samples = len(X)
        n_features = len(X[0])
        self.weights = [0.0 for _ in range(n_features)]
        
        for iteration in range(self.iterations):
            for i in range(n_samples):
                linear_combination = dot_product(X[i], self.weights)
                y_predicted = sigmoid(linear_combination)
                update = [self.learning_rate * (y[i] - y_predicted) * x_i for x_i in X[i]]
                self.weights = [w + u for w, u in zip(self.weights, update)]
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.iterations}")
    
    def predict_prob(self, X):
        return [sigmoid(dot_product(x, self.weights)) for x in X]
    
    def predict(self, X):
        probabilities = self.predict_prob(X)
        predictions = [1 if prob >= 0.5 else 0 for prob in probabilities]
        print(f"Generated predictions for {len(predictions)} samples")
        return predictions

# Evaluation
def evaluate_metrics(y_true, y_pred):
    true_positive = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    true_negative = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    false_positive = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    false_negative = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    
    accuracy = (true_positive + true_negative) / len(y_true)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    return accuracy, precision, recall, f1_score

# Load and preprocess the datasets
train_data = load_csv('./train-1.csv')
test_data = load_csv('./test-1.csv')

X_train, y_train = preprocess_data(train_data)
X_test, y_test = preprocess_data(test_data)

# Initialize and train the Logistic Regression model
model = LogisticRegressionSGD()
print("Starting training...")
model.fit(X_train, y_train)

# Predict on test set
print("Predicting on test set...")
predictions = model.predict(X_test)

# Calculate and print evaluation metrics
accuracy, precision, recall, f1_score = evaluate_metrics(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
