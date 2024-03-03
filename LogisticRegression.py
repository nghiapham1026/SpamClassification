import csv
import math
import matplotlib.pyplot as plt  # Import matplotlib for plotting the loss over iterations.

# Function to load data from a CSV file.
# This function skips the header row and returns the rest of the data.
def load_csv(filepath):
    with open(filepath, 'r') as file:
        csv_reader = csv.reader(file)
        data = list(csv_reader)
    print(f"Loaded {len(data)-1} rows from {filepath}")
    return data[1:]  # Skipping header

# Function to preprocess data.
# Converts feature values to floats and labels to integers.
# Also counts and prints the number of positive and negative instances.
def preprocess_data(data):
    X = [list(map(float, row[:-1])) for row in data]  # Convert feature values to floats.
    y = [int(row[-1]) for row in data]  # Convert label values to integers.
    # Count and print the distribution of classes in the dataset.
    positive_count = sum(label == 1 for label in y)
    negative_count = sum(label == 0 for label in y)
    print(f"Preprocessed data: {len(X)} samples")
    print(f"Positive instances: {positive_count}, Negative instances: {negative_count}")
    return X, y

# Function to calculate the dot product of two vectors.
def dot_product(v1, v2):
    return sum(x*y for x, y in zip(v1, v2))

# Sigmoid activation function used for binary classification.
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# Logistic Regression model using Stochastic Gradient Descent.
class LogisticRegressionSGD:
    def __init__(self, learning_rate=0.01, iterations=200):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None  # Model weights are initialized in the fit method.
    
    # The fit method for training the model.
    # It updates the weights based on the logistic regression gradient descent update rule.
    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        self.weights = [0.0] * n_features  # Initialize weights to zeros.
        self.loss_history = []  # To store the loss at each iteration for plotting.
        # Training loop
        for _ in range(self.iterations):
            current_loss = 0
            for i in range(n_samples):
                linear_combination = dot_product(X[i], self.weights)
                y_predicted = sigmoid(linear_combination)
                # Update rule for logistic regression.
                update = [self.learning_rate * (y[i] - y_predicted) * x_i for x_i in X[i]]
                self.weights = [w + u for w, u in zip(self.weights, update)]
                # Loss calculation for logging.
                current_loss += y[i] * math.log(y_predicted) + (1 - y[i]) * math.log(1 - y_predicted)
            current_loss = -current_loss / n_samples
            self.loss_history.append(current_loss)
            if (_ + 1) % 10 == 0:
                print(f"Iteration {_ + 1}/{self.iterations}, Loss: {current_loss}")
    
    # Predict probabilities of the positive class.
    def predict_prob(self, X):
        return [sigmoid(dot_product(x, self.weights)) for x in X]
    
    # Predict class labels based on probabilities.
    def predict(self, X):
        probabilities = self.predict_prob(X)
        return [1 if prob >= 0.5 else 0 for prob in probabilities]

# Function to evaluate the model on a dataset.
# Calculates and prints accuracy, precision, recall, and F1-score.
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

# Main script starts here.
# Load the training and test datasets.
train_data = load_csv('./train-1.csv')
test_data = load_csv('./test-1.csv')

# Preprocess the datasets to get features and labels.
X_train, y_train = preprocess_data(train_data)
X_test, y_test = preprocess_data(test_data)

# Initialize the Logistic Regression model and train it.
model = LogisticRegressionSGD()
print("Starting training...")
model.fit(X_train, y_train)

# Evaluate the model on the training set.
print("\nEvaluating on training set...")
train_predictions = model.predict(X_train)
evaluate_metrics(y_train, train_predictions)

# Evaluate the model on the test set.
print("\nPredicting on test set...")
test_predictions = model.predict(X_test)
evaluate_metrics(y_test, test_predictions)

# Print the total cost of the model (last log loss value) and plot the loss over iterations.
total_cost = model.loss_history[-1]
print(f"Total cost of the model: {total_cost}")
plt.plot(model.loss_history)
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