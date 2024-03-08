# SpamClassification - Programming Assignment #1
### Image of log loss plot is at plot.png

This project implements a logistic regression model using stochastic gradient descent (SGD) in both Python and Java. The model is trained to classify data into two classes, which are typical for spam detection problems. The scripts will output the evaluation metrics for both the training and test sets, including the total cost of the model as calculated by the final log loss.

## Project Scripts

The project contains two scripts:
- `LogisticRegression.py`: The Python implementation of SGD for logistic regression.
- `LogisticRegressionSGD.java`: The Java implementation of SGD for logistic regression. The Java script is much faster than Python

## How to use

### Python

Install `matplotlib` for plotting the log loss curve:

```bash
pip install matplotlib
```

Run the script using Python 3:

```bash
python LogisticRegression.py
```

### Java

Compile and run the Java script using:

```bash
javac LogisticRegressionSGD.java
java LogisticRegressionSGD
```
