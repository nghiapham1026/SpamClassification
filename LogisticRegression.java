
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
public class LogisticRegression {
/** the learning rate */
private double rate=0.01;
/** the weights to learn */
private double[] weights;
/** the number of iterations */
private int ITERATIONS = 200;
/* Constructor initializes the weight vector. Initialize it by setting it
to the 0 vector. **/
/* Implement the sigmoid function **/
/* Helper function for prediction **/
/** Takes a test instance as input and outputs the probability of the label
being 1 **/
/** This function should call sigmoid() **/
/* The prediction function **/
/** Takes a test instance as input and outputs the predicted label **/
/** This function should call Helper function **/
/** This function takes a test set as input, call the predict function to
predict a label for it, **/
/** and prints the accuracy, P, R, and F1 score of the positive class and
negative class and the confusion matrix **/
/** Train the Logistic Regression in a function using Stochastic Gradient
Descent **/
/** Also compute the log-oss in this function **/
/** Function to read the input dataset **/
/** main Function **/
}
