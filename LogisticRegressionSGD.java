import java.io.*;
import java.util.ArrayList;
import java.util.List;

// Define the LogisticRegressionSGD class.
public class LogisticRegressionSGD {
    // Model parameters: learning rate, number of iterations, weights for the features, and history of log loss values.
    private double learningRate;
    private int iterations;
    private double[] weights;
    private List<Double> logLossHistory;

    // Constructor to initialize the model with a specific learning rate and number of iterations.
    public LogisticRegressionSGD(double learningRate, int iterations) {
        this.learningRate = learningRate;
        this.iterations = iterations;
        this.weights = null; // Weights are initialized during training.
        this.logLossHistory = new ArrayList<>(); // To track log loss over iterations.
    }

    // Method to load data from a CSV file.
    public static List<String[]> loadCsv(String filepath) {
        List<String[]> data = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filepath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                data.add(values);
            }
            System.out.println("Loaded " + (data.size() - 1) + " rows from " + filepath);
        } catch (IOException e) {
            e.printStackTrace();
        }
        data.remove(0); // Remove the header row.
        return data;
    }

    // Calculate the dot product of two vectors.
    public static double dotProduct(double[] v1, double[] v2) {
        double sum = 0;
        for (int i = 0; i < v1.length; i++) {
            sum += v1[i] * v2[i];
        }
        return sum;
    }

    // Sigmoid function used for binary classification.
    public static double sigmoid(double z) {
        return 1 / (1 + Math.exp(-z));
    }

    // Compute log loss for a set of predictions.
    private double logLoss(int[] y, double[] probabilities) {
        double loss = 0.0;
        for (int i = 0; i < y.length; i++) {
            loss += y[i] * Math.log(probabilities[i]) + (1 - y[i]) * Math.log(1 - probabilities[i]);
        }
        return -loss / y.length;
    }

    // Train the logistic regression model using SGD.
    public void fit(double[][] X, int[] y) {
        int nSamples = X.length;
        int nFeatures = X[0].length;
        this.weights = new double[nFeatures];

        for (int iteration = 0; iteration < this.iterations; iteration++) {
            double[] probabilities = new double[nSamples];
            for (int i = 0; i < nSamples; i++) {
                double linearCombination = dotProduct(X[i], this.weights);
                probabilities[i] = sigmoid(linearCombination);
                for (int j = 0; j < nFeatures; j++) {
                    this.weights[j] += this.learningRate * (y[i] - probabilities[i]) * X[i][j];
                }
            }
            double iterationLogLoss = logLoss(y, probabilities);
            logLossHistory.add(iterationLogLoss);

            if ((iteration + 1) % 10 == 0 || iteration == this.iterations - 1) {
                System.out.println("Iteration " + (iteration + 1) + "/" + this.iterations + " - Log Loss: " + iterationLogLoss);
            }
        }
    }

    // Predict probabilities of being in the positive class.
    public double[] predictProb(double[][] X) {
        double[] probabilities = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            probabilities[i] = sigmoid(dotProduct(X[i], this.weights));
        }
        return probabilities;
    }

    // Predict class labels based on probabilities.
    public int[] predict(double[][] X) {
        double[] probabilities = predictProb(X);
        int[] predictions = new int[probabilities.length];
        for (int i = 0; i < probabilities.length; i++) {
            predictions[i] = probabilities[i] >= 0.5 ? 1 : 0;
        }
        System.out.println("Generated predictions for " + predictions.length + " samples");
        return predictions;
    }

    // Preprocess features from the dataset.
    public static double[][] preprocessFeatures(List<String[]> data) {
        double[][] features = new double[data.size()][data.get(0).length - 1];
        for (int i = 0; i < data.size(); i++) {
            for (int j = 0; j < data.get(i).length - 1; j++) {
                features[i][j] = Double.parseDouble(data.get(i)[j]);
            }
        }
        return features;
    }

    // Preprocess labels from the dataset.
    public static int[] preprocessLabels(List<String[]> data) {
        int[] labels = new int[data.size()];
        for (int i = 0; i < data.size(); i++) {
            labels[i] = Integer.parseInt(data.get(i)[data.get(i).length - 1]);
        }
        return labels;
    }

    // Evaluate model metrics and print them.
    public static void evaluateMetrics(int[] yTrue, int[] yPred, String setName) {
        // Calculate metrics such as true positives, true negatives, etc.
        int truePositive = 0;
        int trueNegative = 0;
        int falsePositive = 0;
        int falseNegative = 0;
        for (int i = 0; i < yTrue.length; i++) {
            if (yTrue[i] == 1 && yPred[i] == 1) truePositive++;
            else if (yTrue[i] == 0 && yPred[i] == 0) trueNegative++;
            else if (yTrue[i] == 0 && yPred[i] == 1) falsePositive++;
            else if (yTrue[i] == 1 && yPred[i] == 0) falseNegative++;
        }
        // Compute accuracy, precision, recall, and F1 score.
        double accuracy = (double) (truePositive + trueNegative) / (yTrue.length);
        double precisionPositive = truePositive / (double) (truePositive + falsePositive);
        double recallPositive = truePositive / (double) (truePositive + falseNegative);
        double f1ScorePositive = 2 * (precisionPositive * recallPositive) / (precisionPositive + recallPositive);

        double precisionNegative = trueNegative / (double) (trueNegative + falseNegative);
        double recallNegative = trueNegative / (double) (trueNegative + falsePositive);
        double f1ScoreNegative = 2 * (precisionNegative * recallNegative) / (precisionNegative + recallNegative);

        // Print the metrics.
        System.out.println("----- " + setName + " Set Metrics -----");
        System.out.println("Accuracy: " + accuracy);
        System.out.println("Positive Class (Spam) Precision: " + precisionPositive);
        System.out.println("Positive Class (Spam) Recall: " + recallPositive);
        System.out.println("Positive Class (Spam) F1 Score: " + f1ScorePositive);
        System.out.println("Negative Class (Ham) Precision: " + precisionNegative);
        System.out.println("Negative Class (Ham) Recall: " + recallNegative);
        System.out.println("Negative Class (Ham) F1 Score: " + f1ScoreNegative);
        System.out.println("Confusion Matrix:");
        System.out.println("\tTrue Negative (Ham): " + trueNegative);
        System.out.println("\tFalse Positive (Spam as Ham): " + falsePositive);
        System.out.println("\tFalse Negative (Ham as Spam): " + falseNegative);
        System.out.println("\tTrue Positive (Spam): " + truePositive);
    }

    // Main method to execute the program.
    public static void main(String[] args) {
        // Load and preprocess the training and testing datasets.
        List<String[]> trainData = loadCsv("./train-1.csv");
        List<String[]> testData = loadCsv("./test-1.csv");
    
        // Extract features and labels from the datasets.
        double[][] XTrain = preprocessFeatures(trainData);
        int[] yTrain = preprocessLabels(trainData);
        double[][] XTest = preprocessFeatures(testData);
        int[] yTest = preprocessLabels(testData);
    
        // Initialize and train the logistic regression model.
        LogisticRegressionSGD model = new LogisticRegressionSGD(0.01, 200);
        System.out.println("Starting training...");
        model.fit(XTrain, yTrain);
    
        // Make predictions on the training and test sets, and evaluate the model.
        System.out.println("Predicting on test set...");
        int[] trainPredictions = model.predict(XTrain);
        evaluateMetrics(yTrain, trainPredictions, "Train");
        int[] testPredictions = model.predict(XTest);
        evaluateMetrics(yTest, testPredictions, "Test");
    
        // Print the final log loss of the model as an indication of total cost.
        double totalCost = model.logLossHistory.get(model.logLossHistory.size() - 1);
        System.out.println("Total cost of the model: " + totalCost);
    }    
}

/*
Loaded 4459 rows from ./train-1.csv
Loaded 1115 rows from ./test-1.csv
Positive instances in train file: 586
Negative instances in train file: 3873
Starting training...
Iteration 10/200 - Log Loss: 0.1750429972941354
Iteration 20/200 - Log Loss: 0.13750056938544888
Iteration 30/200 - Log Loss: 0.11861972786595595
Iteration 40/200 - Log Loss: 0.10625013527390886
Iteration 50/200 - Log Loss: 0.09717340275983574
Iteration 60/200 - Log Loss: 0.09009117550282754
Iteration 70/200 - Log Loss: 0.08434660412992144
Iteration 80/200 - Log Loss: 0.07955900052779098
Iteration 90/200 - Log Loss: 0.07548798031526856
Iteration 100/200 - Log Loss: 0.07197228837238939
Iteration 110/200 - Log Loss: 0.06889821038663452
Iteration 120/200 - Log Loss: 0.06618213965002254
Iteration 130/200 - Log Loss: 0.06376077173897395
Iteration 140/200 - Log Loss: 0.06158525874655854
Iteration 150/200 - Log Loss: 0.05961734324323422
Iteration 160/200 - Log Loss: 0.05782662206892225
Iteration 170/200 - Log Loss: 0.056188577164825934
Iteration 180/200 - Log Loss: 0.05468316030958641
Iteration 190/200 - Log Loss: 0.05329377554796852
Iteration 200/200 - Log Loss: 0.0520065413616742
Predicting on test set...
Generated predictions for 4459 samples
----- Train Set Metrics -----
Accuracy: 0.9818344920385736
Positive Class (Spam) Precision: 0.8902627511591963
Positive Class (Spam) Recall: 0.9829351535836177
Positive Class (Spam) F1 Score: 0.9343065693430657
Negative Class (Ham) Precision: 0.9973767051416579
Negative Class (Ham) Recall: 0.9816679576555641
Negative Class (Ham) F1 Score: 0.9894599869876382
Confusion Matrix:
        True Negative (Ham): 3802
        False Positive (Spam as Ham): 71
        False Negative (Ham as Spam): 10
        True Positive (Spam): 576
Generated predictions for 1115 samples
----- Test Set Metrics -----
Accuracy: 0.9560538116591928
Positive Class (Spam) Precision: 0.8111111111111111
Positive Class (Spam) Recall: 0.906832298136646
Positive Class (Spam) F1 Score: 0.8563049853372435
Negative Class (Ham) Precision: 0.983957219251337
Negative Class (Ham) Recall: 0.9643605870020965
Negative Class (Ham) F1 Score: 0.9740603493912123
Confusion Matrix:
        True Negative (Ham): 920
        False Positive (Spam as Ham): 34
        False Negative (Ham as Spam): 15
        True Positive (Spam): 146
Total cost of the model: 0.0520065413616742
 */